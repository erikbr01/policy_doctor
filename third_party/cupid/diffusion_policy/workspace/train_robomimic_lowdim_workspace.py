if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import math
import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_lowdim_policy import RobomimicLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.device_util import get_device, non_blocking_for


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainRobomimicLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicLowdimPolicy = hydra.utils.instantiate(cfg.policy)

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
            train_sampler = None
            train_dataloader = DataLoader(dataset, **cfg.dataloader)

        normalizer = dataset.get_normalizer()

        # configure validation dataset (no sampler — rank 0 runs full val)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env (main rank only to avoid parallel sim instances)
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

        # device transfer
        self.model.to(device)

        # compile before DDP wrap (torch 2.x recommendation)
        if use_compile:
            from diffusion_policy.common.ddp_util import compile_model
            self.model = compile_model(self.model)

        # DDP wrap (note: robomimic policy manages its own optimizer internally)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[rank])

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1

        # If num_gradient_steps is set, compute num_epochs from it (steps per epoch = len(dataloader))
        num_gradient_steps = cfg.training.get("num_gradient_steps")
        if num_gradient_steps is not None and num_gradient_steps > 0:
            steps_per_epoch = len(train_dataloader)
            if steps_per_epoch > 0:
                cfg.training.num_epochs = math.ceil(num_gradient_steps / steps_per_epoch)
                print(
                    f"Training for {num_gradient_steps} gradient steps "
                    f"({cfg.training.num_epochs} epochs × {steps_per_epoch} steps/epoch)"
                )

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with (JsonLogger(log_path) if is_main else open(os.devnull, 'w')) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # logging
                        loss_cpu = info['losses']['action_loss'].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if is_main:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch (main rank only) ==========
                if is_main:
                    self.model.eval()

                    # run rollout
                    if (self.epoch % cfg.training.rollout_every) == 0:
                        runner_log = env_runner.run(self.model)
                        # log all
                        step_log.update(runner_log)

                    # run validation
                    if (self.epoch % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=non_blocking_for(device)))
                                    info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                    loss = info['losses']['action_loss']
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                    # checkpoint
                    if (self.epoch % cfg.training.checkpoint_every) == 0:
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
                    self.model.train()

                    # end of epoch
                    # log of last step is combined with validation and rollout
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRobomimicLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
