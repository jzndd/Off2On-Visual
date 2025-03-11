import os
from pathlib import Path
from typing import List, Union
from mw_wrapper import TorchEnv, make_mw_env
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from trainer2D import Trainer
os.environ["MUJOCO_GL"] = "egl"


def run(cfg, root_dir):
    trainer = Trainer(cfg, root_dir)
    trainer.run()

@hydra.main(config_path="config/", config_name="trainer2D", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    root_dir = Path(hydra.utils.get_original_cwd())

    if cfg.debug_mode:
        cfg.wandb.mode = "disabled"         # If debug mode, disable use of wandb
        cfg.collection.train.first_epoch.min = 1500
        # For ActionModel
        cfg.actor_critic.training.batch_size = 256
        cfg.actor_critic.training.steps_per_epoch = 10
        cfg.actor_critic.training.steps_first_epoch = 10
        #
        cfg.training.bc_actor_warmup_steps = 100
        cfg.training.bc_critic_warmup_steps = 100
        cfg.training.online_max_iter = 2000
        cfg.evaluation.every_iter = 1000

    run(cfg, root_dir)

def setup_visible_cuda_devices(devices: Union[str, int, List[int]]) -> None:
    if isinstance(devices, str):
        raise NotImplementedError("Only one gpu device is supported")
        if devices == "cpu":
            devices = []
        else:
            assert devices == "all"
            return
    elif isinstance(devices, int):
        devices = [devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))

if __name__ == "__main__":
    main()