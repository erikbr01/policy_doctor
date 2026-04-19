if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.device_util import get_device, non_blocking_for
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # --- acceleration flags (DDP / TF32 / compile) ---
        rank        = int(os.environ.get("LOCAL_RANK", cfg.training.get("_ddp_rank", 0)))
        world_size  = int(os.environ.get("WORLD_SIZE",  cfg.training.get("_ddp_world_size", 1)))
        is_main     = (rank == 0)
        use_tf32    = bool(cfg.training.get("tf32", False))
        use_compile = bool(cfg.training.get("compile", False))

        if world_size > 1:
            from diffusion_policy.common.ddp_util import setup_ddp
            setup_ddp(rank=rank, world_size=world_size)

        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)

        # device: DDP uses one GPU per rank; single-GPU falls back to cfg
        if world_size > 1:
            device = torch.device(f"cuda:{rank}")
        else:
            device = get_device(cfg.training.device)

        # train dataloader: DistributedSampler for DDP, standard otherwise
        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            train_dl_kw = {k: v for k, v in cfg.dataloader.items() if k != "shuffle"}
            train_dataloader = DataLoader(dataset, sampler=train_sampler, **train_dl_kw)
        else:
            train_dl_kw = dict(cfg.dataloader)
            if device.type == "mps":
                train_dl_kw["pin_memory"] = False
            train_dataloader = DataLoader(dataset, **train_dl_kw)

        normalizer = dataset.get_normalizer()

        # configure validation dataset (no sampler — rank 0 runs full val)
        val_dataset = dataset.get_validation_dataset()
        val_dl_kw = dict(cfg.val_dataloader)
        if not world_size > 1 and device.type == "mps":
            val_dl_kw["pin_memory"] = False
        val_dataloader = DataLoader(val_dataset, **val_dl_kw)

        getattr(self.model, "module", self.model).set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # training duration: num_steps overrides num_epochs when set
        import math
        _num_steps = cfg.training.get("num_steps", None)
        _steps_per_epoch = len(train_dataloader)
        num_epochs_to_run = (
            math.ceil(_num_steps / _steps_per_epoch)
            if _num_steps is not None
            else cfg.training.num_epochs
        )

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                (_num_steps if _num_steps is not None else _steps_per_epoch * num_epochs_to_run)
                // cfg.training.gradient_accumulate_every),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner (only on main rank to avoid parallel sim instances)
        env_runner: BaseLowdimRunner = None
        if is_main:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging (main rank only)
        wandb_run = None
        if is_main:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update({"output_dir": self.output_dir})

        # configure checkpoint (main rank only)
        topk_manager = None
        if is_main:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # device transfer (CUDA / MPS / CPU)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # compile before DDP wrap (torch 2.x recommendation)
        if use_compile:
            from diffusion_policy.common.ddp_util import compile_model
            self.model = compile_model(self.model)

        # DDP wrap
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[rank])

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            num_epochs_to_run = 2
            _num_steps = None
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        terminate_training = False
        with (JsonLogger(log_path) if is_main else open(os.devnull, 'w')) as json_logger:
            for local_epoch_idx in range(num_epochs_to_run):
                if terminate_training:
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                if world_size > 1:
                    train_sampler.set_epoch(self.epoch)
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = getattr(self.model, "module", self.model).compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            from diffusion_policy.common.ddp_util import unwrap_model; ema.step(unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1
                            if _num_steps is not None and self.global_step >= _num_steps:
                                terminate_training = True
                                break

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch (main rank only) ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                if is_main:
                    policy.eval()

                    # run rollout
                    if (self.epoch % cfg.training.rollout_every) == 0 or terminate_training:
                        runner_log = env_runner.run(policy)
                        # log all
                        step_log.update(runner_log)

                    # run validation
                    if (self.epoch % cfg.training.val_every) == 0 or terminate_training:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                                    loss = getattr(self.model, "module", self.model).compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                    # run diffusion sampling on a training batch
                    if (self.epoch % cfg.training.sample_every) == 0 or terminate_training:
                        with torch.no_grad():
                            # sample trajectory from training set, and evaluate difference
                            batch = train_sampling_batch
                            obs_dict = {'obs': batch['obs']}
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            if cfg.pred_action_steps_only:
                                pred_action = result['action']
                                start = cfg.n_obs_steps - 1
                                end = start + cfg.n_action_steps
                                gt_action = gt_action[:,start:end]
                            else:
                                pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            # log
                            step_log['train_action_mse_error'] = mse.item()
                            # release RAM
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse

                    # checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0 or terminate_training:
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                if is_main:
                    policy.train()

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                if terminate_training or (
                    _num_steps is not None and self.global_step >= _num_steps
                ):
                    break

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
