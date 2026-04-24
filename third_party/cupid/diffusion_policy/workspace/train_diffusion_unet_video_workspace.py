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
import random
import time
import wandb
import tqdm
import numpy as np

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_video_policy import DiffusionUnetVideoPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.device_util import get_device, non_blocking_for
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.common.step_timer import StepTimer

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetVideoWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetVideoPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetVideoPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
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
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

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
            train_sampler = None
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

        normalizer = dataset.get_normalizer()

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
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

        # configure env (main rank only to avoid parallel sim instances)
        env_runner: BaseImageRunner = None
        if is_main:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

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

        # device transfer
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

        # save batch for validation (from rank-local dataloader)
        val_batch = next(iter(train_dataloader))

        step_timer = StepTimer()

        # training loop
        for _ in range(cfg.training.num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(self.epoch)
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch in tepoch:
                    step_timer.reset()
                    # device transfer
                    with step_timer.time("data_transfer"):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))

                    # compute loss
                    with step_timer.time("forward_backward"):
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        with step_timer.time("optimizer_step"):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                    # update ema
                    if cfg.training.use_ema:
                        with step_timer.time("ema_update"):
                            ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(step_timer.to_log_dict())

                    # eval (main rank only)
                    if is_main:
                        should_eval = self.global_step % cfg.training.eval_every == 0
                        if (not cfg.training.eval_first) and (self.global_step == 0):
                            should_eval = False
                        if should_eval:
                            policy = self.model
                            if cfg.training.use_ema:
                                policy = self.ema_model
                            policy.eval()
                            _t0 = time.perf_counter()
                            runner_log = env_runner.run(policy)
                            step_log['timer/rollout'] = time.perf_counter() - _t0
                            policy.train()
                            step_log.update(runner_log)

                            # checkpointing
                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint()
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot()

                            save_path = topk_manager.get_ckpt_path({
                                'epoch': self.epoch,
                                'test_score': runner_log['test/mean_score']
                            })
                            if save_path is not None:
                                self.save_checkpoint(path=save_path)

                        # validation
                        should_val = self.global_step % cfg.training.val_every == 0
                        if should_val:
                            policy = self.model
                            if cfg.training.use_ema:
                                policy = self.ema_model
                            policy.eval()
                            _t0 = time.perf_counter()
                            with torch.no_grad():
                                # sample trajectory from training set, and evaluate difference
                                batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                                obs_dict = batch['obs']
                                gt_action = batch['action']

                                result = policy.predict_action(obs_dict)
                                pred_action = result['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                step_log['train_action_mse_error'] = mse.item()
                                del batch
                                del obs_dict
                                del gt_action
                                del result
                                del pred_action
                                del mse
                            step_log['timer/sampling'] = time.perf_counter() - _t0
                            policy.train()

                        # log
                        wandb_run.log(step_log, step=self.global_step)

                    self.global_step += 1
            self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetVideoWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
