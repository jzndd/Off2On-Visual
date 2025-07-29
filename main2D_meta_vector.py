import os
from pathlib import Path
from typing import List, Union
from omegaconf import DictConfig, OmegaConf
import hydra

# os.environ['WANDB_BASE_URL']="https://api.wandb-cn.top"
os.environ['WANDB_API_KEY']="338a1a94f799ce9a1470532e392f97fe330af917"

def run(cfg, root_dir):
    # from trainer2D_meta_vector import Trainer
    # MODIFIED !!!!!!!!!!!!!
    from trainer2D_meta_vector_discrete import Trainer
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
    
    if not cfg.save_data and not cfg.save_reward_data and not cfg.save_multi_data:
        if not cfg.is_sparse_reward:
            cfg.wandb.group += "_wosparse"
        # MODIFIED !!!!!!!!!!!!!
        cfg.wandb.group += f"_discrete_bin{cfg.bin}"
        run(cfg, root_dir)
    elif cfg.save_data and not cfg.save_reward_data:
        from trainer2D_meta_vector_discrete import Trainer
        cfg.wandb.mode = "disabled"
        # cfg.env.test.whole_traj = True
        trainer = Trainer(cfg, root_dir)
        save_whole_traj = False
        # trainer.save_data(save_whole_traj=save_whole_traj)
        trainer.save_eval_data(save_whole_traj=save_whole_traj)
    elif not cfg.save_data and cfg.save_reward_data:
        from trainer2D_meta_vector_discrete import Trainer
        cfg.wandb.mode = "disabled"
        # cfg.env.test.whole_traj = True
        trainer = Trainer(cfg, root_dir)
        trainer.save_reward_data()
        trainer.save_reward_data(np_seed=42, test_mode=True)
    elif cfg.save_multi_data:
        cfg.wandb.mode = "disabled"
        from trainer2D_meta_vector_discrete import Trainer
        trainer = Trainer(cfg, root_dir)
        trainer.save_multi_data()
        trainer.save_multi_eval_data()

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
    os.environ["MUJOCO_EGL_DEVICE_ID"] = ",".join(map(str, devices))

if __name__ == "__main__":
    main()