import os
from pathlib import Path
import random
from typing import List, Tuple, Union
from mw_wrapper import TorchEnv, make_mw_env
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from rl_agent import PPOAgent2D
import wandb
import numpy as np
from rl_agent.utils import OfflineReplaybuffer, OnlineReplayBuffer

from utils import VideoRecorder, wandb_log

class Trainer:

    def __init__(self, cfg, root_dir):
        self._device = torch.device("cuda:0")
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Starting on {self._device}")

        # wandb
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True), 
            reinit=True, 
            resume=True, 
            **cfg.wandb
        )

        self.iter = 0

        # self._path_video_dir = os.path.join(root_dir, "videos")
        # os.makedirs(self._path_video_dir, exist_ok=True)
        self._path_video_dir = Path("videos")
        os.makedirs(self._path_video_dir, exist_ok=True)
        self._save_path = Path("saved_models")
        self._save_path.mkdir(parents=True, exist_ok=True)

        # init agent
        self.agent: PPOAgent2D = PPOAgent2D(cfg.agent.actor_critic_cfg, cfg.agent.ppo_cfg)
        self.agent.to(self._device)
        self.agent.setup_training(cfg.actor_critic.actor_critic_loss)

        self.best_success_rate = -0.01

    def run(self):
        cfg = self._cfg
        train_env = make_mw_env(num_envs=cfg.collection.train.num_envs, device=self._device, **cfg.env.train)
    
        max_iter = self._cfg.training.online_max_iter
        bc_warmup_steps = self._cfg.training.bc_warmup_steps

        if self._cfg.expert_rb_dir is not None:
            expertrb = OfflineReplaybuffer(capacity=5000)
            expertrb.load(self._cfg.expert_rb_dir)
            expertrb.compute_returns()

        # assert self._cfg.actor_critic.training.batch_size >= 1024  # PPO's batch_size >= 1024
        rb = OnlineReplayBuffer(self._cfg.actor_critic.training.batch_size, (3, 84, 84), 4)

        eps_rew = 0
        done = False
        to_log = []
        step = 0

        while self.iter < max_iter:

            self.iter += 1

            # -----------------------------   2-stage params update   ----------------------------- 
            # 1 stage : bc warm up
            if self.iter == 1:
                for i in range(bc_warmup_steps):
                    metrics = self.agent.bc_update(expertrb)

                to_log += self.test_actor_critic()
                self.save_agents("bc", to_log[0]["actor_critic/test/success_rate"])
                self.agent.bc_transfer_ac() 

                obs, _ = train_env.reset()

            # 2 stage : ac update
            with torch.no_grad():
                real_act_tuple = self.agent.predict_act(obs) 
                if isinstance(real_act_tuple, Tuple):
                    real_act = real_act_tuple[0]
                    old_log_prob = real_act_tuple[1]
                    if len(real_act_tuple) == 3:
                        state_value = real_act_tuple[2]
                    else:
                        state_value = None
                else:
                    real_act = real_act_tuple
                    old_log_prob = None

                next_obs, rew, end, trunc, info = train_env.step(real_act)
                step += 1
                eps_rew += rew
                done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
                done = done or trunc or end
                rb.store(obs, next_obs, rew, done, real_act, old_log_prob, state_value)

            if done or trunc or end:
                obs, _ = train_env.reset()
                print("this traj eps rew is {}, and the traj len is {}".format(eps_rew, step))
                eps_rew = 0
                step = 0

            obs = next_obs

            if rb.size >= self._cfg.actor_critic.training.batch_size:
                print(" ---------------------- begin update  ------------------".format(rb.size))
                metrics = self.agent.update(rb)
                # if self.iter <= self._cfg.iter_train_world_model:
                #     metrics = self.agent.actor_critic.update(rb, expertrb)
                # else:
                #     metrics = self.agent.actor_critic.update(rb, None)

                _to_log = []
                _to_log.append(metrics)
                _to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in _to_log]
                to_log += _to_log
                rb.clear()

            should_test = self._cfg.evaluation.should and (self.iter % self._cfg.evaluation.every_iter == 0)
            if should_test:
                print("begin test")
                to_log += self.test_actor_critic()

            wandb_log(to_log, self.iter)
            to_log = []

    @torch.no_grad()
    def test_actor_critic(self, eval_times=25):
        test_env = make_mw_env(num_envs=self._cfg.collection.test.num_envs, device=self._device, **self._cfg.env.test)
        total_reward = 0.0
        success_rate = 0.0
        video_recorder = VideoRecorder(self._path_video_dir)
        for i in range(eval_times):
            test_env.reset()
            done = False
            # hx = torch.zeros(test_env.num_envs, model.lstm_dim, device=model.device)
            # cx = torch.zeros(test_env.num_envs, model.lstm_dim, device=model.device)

            seed = random.randint(0, 2**31 - 1)
            obs, _ = test_env.reset(seed=[seed + i for i in range(test_env.num_envs)])
            enabled=True if i==eval_times-1 else False
            video_recorder.init(obs, enabled=enabled)
            success = False
            steps = 0
            # hx, cx = hx.detach(), cx.detach()

            while True:
                # logits_act, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
                act = self.agent.predict_act(obs, eval_mode=True)
                # import pdb; pdb.set_trace()
                # act_matrix = segmented_activation(logits_act)
                # actual_act, act_matrix = decode_metaworld_action(act_matrix, return_index_matrix=True)  # [1, 84] -> [1, 4], [1, 84]

                # next_obs, rew, end, trunc, info = test_env.step(actual_act)
                next_obs, rew, end, trunc, info = test_env.step(act)
                video_recorder.record(next_obs)
                # import pdb; pdb.set_trace()
                # video_recorder.save(f"{self.epoch}.mp4")
                steps += 1
                total_reward += rew.sum().item()

                success |= bool(info['success'])

                obs = next_obs

                if end or trunc:
                    break

            if success:
                success_rate += 1

            video_recorder.save(f"{self.iter}.mp4")
            print("the {} traj success is {} and steps is {}".format(i, success, steps))


        success_rate = success_rate / eval_times
        avg_reward = total_reward / eval_times

        # 如果存在 set_old_policy 方法，且成功率大于最佳成功率，则更新最佳成功率
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.save_agents("best", self.best_success_rate)
            if hasattr(self.agent, "set_old_policy"):
                self.agent.set_old_policy()

        to_log = [{"success_rate": success_rate, "avg_reward": avg_reward}]
        # import pdb; pdb.set_trace()
        # T H W C -> T C H W
        save_frames = np.array(video_recorder.frames).transpose(0, 3, 1, 2)
        video_wandb = wandb.Video(save_frames, fps=video_recorder.fps, format="mp4")
        to_log.append({"video": video_wandb})

        print(f"Success rate: {success_rate}, Average reward: {avg_reward}")
        to_log = [{f"actor_critic/test/{k}": v for k, v in d.items()} for d in to_log]
        print("end test actor_critic")

        return to_log

    def save_agents(self, name="best", score=None):
        """Saves the offline agent and best agent."""
        if score is None:
            agent_path = self._save_path / f"{name}.pth"
        else:
            score = str(int(100 * score))
            agent_path = self._save_path / f"{name}_{score}.pth"

        # Save the offline agent
        torch.save({
            "agent": self.agent.actor.state_dict(),
        }, agent_path)
