import os
from pathlib import Path
from typing import List, Union
from omegaconf import DictConfig, OmegaConf
import hydra
os.environ["MUJOCO_GL"] = "egl"

# os.environ['WANDB_BASE_URL']="https://api.wandb-cn.top"
os.environ['WANDB_API_KEY']="338a1a94f799ce9a1470532e392f97fe330af917"

def run(cfg, root_dir):
    from trainer2D_maniskill_vector_discrete import Trainer
    trainer = Trainer(cfg, root_dir)
    trainer.run()

@hydra.main(config_path="config/", config_name="trainer2D_maniskill", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_visible_cuda_devices(cfg.common.devices)
    root_dir = Path(hydra.utils.get_original_cwd())

    if cfg.debug_mode:
        cfg.wandb.mode = "disabled"         # If debug mode, disable use of wandb
        cfg.collection.train.first_epoch.min = 1500
        # For ActionModel
        cfg.actor_critic.training.steps_per_epoch = 10
        cfg.actor_critic.training.steps_first_epoch = 10
        #
        cfg.training.bc_actor_warmup_steps = 10
        cfg.training.bc_critic_warmup_steps = 10
        # cfg.training.online_max_iter = 2000
        cfg.evaluation.every_iter = 2000
        cfg.evaluation.eval_times = 5

    if cfg.only_bc:
        cfg.wandb.mode = "disabled"         # If debug mode, disable use of wandb
    
    if not cfg.save_data and not cfg.save_reward_data:
        if not cfg.is_sparse_reward:
            cfg.wandb.group += "_wosparse"
        run(cfg, root_dir)
    elif cfg.save_data and not cfg.save_reward_data:
        cfg.wandb.mode = "disabled"
        from trainer2D_maniskill_vector_discrete import Trainer
        trainer = Trainer(cfg, root_dir)
        trainer.save_data()
        trainer.save_eval_data()
    elif not cfg.save_data and cfg.save_reward_data:
        cfg.wandb.mode = "disabled"
        from trainer2D_maniskill_vector_discrete import Trainer
        trainer = Trainer(cfg, root_dir)
        # trainer.save_reward_data()
        trainer.save_reward_data(np_seed=42, test_mode=True)
    else:
        raise NotImplementedError("Only save_data and save_reward_data are supported")

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