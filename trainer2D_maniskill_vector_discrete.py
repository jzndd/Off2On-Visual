import os
from pathlib import Path
import random
from typing import List, Tuple, Union

import cv2
from mani_wrapper import make_mani_env
# from mw_wrapper import make_mw_env
# from dmc_wrapper_state import get_d4rl_env
import torch
from omegaconf import DictConfig, OmegaConf
from rl_agent import PPOAgent2D
import wandb
import numpy as np
from rl_agent.utils import OfflineReplaybuffer, OnlineReplayBuffer, VectorOnlineReplayBuffer

from utils import VideoRecorder, wandb_log, Normalization, to_np
import sys

from copy import deepcopy

class Trainer:

    def __init__(self, cfg, root_dir):

        sys.path.append(str(root_dir))

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

        self.registered_env_func = make_mani_env
        self.domain_name = 'maniskill'

        self.test_env = self.registered_env_func(device=self._device, **cfg.env.test)
        cfg.agent.actor_critic_cfg.num_actions = deepcopy(self.test_env.num_actions)
        # cfg.agent.actor_critic_cfg.num_states = deepcopy(env.num_states)

        # self.test_env = self.registered_env_func(num_envs=1, device=self._device, **cfg.env.test)
        # obs, info = self.test_env.reset()
        # init agent
        self.agent: PPOAgent2D = PPOAgent2D(cfg.agent.actor_critic_cfg, cfg.agent.ppo_cfg)
        self.agent.to(self._device)
        self.agent.setup_training(cfg.actor_critic.actor_critic_loss)

        self.metrics = -0.01

        if cfg.is_sparse_reward:
            cfg.bc_ckpt_dir = cfg.bc_ckpt_dir.replace("/bc", "_sparse/bc")
        
        if cfg.is_whole_traj:
            cfg.bc_ckpt_dir = cfg.bc_ckpt_dir.replace("/bc", "_wholetraj/bc")

        self.bc_ckpt_dir = cfg.bc_ckpt_dir
        self._cfg = cfg

        self.best_success_rate = -0.01

    def run(self):
        cfg = self._cfg
        train_env = self.registered_env_func(device=self._device, **cfg.env.train)
    
        max_iter = self._cfg.training.online_max_iter
        bc_actor_warmup_steps = self._cfg.training.bc_actor_warmup_steps
        bc_critic_warmup_steps = self._cfg.training.bc_critic_warmup_steps

        # 
        num_envs = train_env.num_envs

        # assert self._cfg.actor_critic.training.batch_size >= 1024  # PPO's batch_size >= 1024
        rb = VectorOnlineReplayBuffer(num_envs, self._cfg.actor_critic.training.batch_size, train_env.observation_space.shape[1:], (train_env.action_space.shape[1],))

        to_log = []

        # First stage: BC
        if self._cfg.train_with_bc:
            if self._cfg.expert_rb_dir is not None:
                if self._cfg.is_sparse_reward:
                    self._cfg.expert_rb_dir = self._cfg.expert_rb_dir.replace("reward", "reward_sparse")
                if self._cfg.is_whole_traj:
                    self._cfg.expert_rb_dir = self._cfg.expert_rb_dir.replace(".pkl", "_whole_traj_50traj.pkl")
                
                expertrb = OfflineReplaybuffer(5000, train_env.observation_space.shape[1:], (train_env.action_space.shape[1],))
                expertrb.load(self._cfg.expert_rb_dir)
                expertrb.compute_returns()
            
            if os.path.exists(self.bc_ckpt_dir):
                ckpt = torch.load(self.bc_ckpt_dir, self._device)
                self.agent.load_state_dict(ckpt["agent"])
                print("load bc ckpt from {}".format(self.bc_ckpt_dir))
                self.test_actor_critic(self._cfg.evaluation.eval_times)
            else:
                for i in range(bc_actor_warmup_steps):
                    metrics = self.agent.bc_actor_update(expertrb)
                    if (i+1) % 10000 == 1:
                        print(f"bc_actor_warmup_steps: {i}")
                        self.test_actor_critic(eval_times=10)

                for i in range(bc_critic_warmup_steps):
                    metrics = self.agent.bc_critic_update(expertrb)

                to_log += self.test_actor_critic(self._cfg.evaluation.eval_times)
                self.save_agents("bc", to_log[0]["actor_critic/test/avg_reward"])

                os.makedirs(os.path.dirname(self.bc_ckpt_dir), exist_ok=True)
                torch.save({
                    "agent": self.agent.state_dict(),
                }, self.bc_ckpt_dir)

        # ckpt = torch.load("/data0/jzn/workspace/Off2On-Visual/outputs/2025.05.15/051913_PushCube-v1_ppo_gae_wholetraj_usebc_use50traj_seed1000_bs2048_minibs64/saved_models/best_86.pth", self._device)
        # self.agent.load_state_dict(ckpt['agent'])
        self.agent.bc_transfer_ac()

        if self._cfg.only_bc:
            exit(1) 

        singe_batch_iter = rb.capacity // num_envs
        obs, _ = train_env.reset(seed=[random.randint(0, 2**31 - 1) for _ in range(num_envs)])
        done = torch.zeros((num_envs, ), device=self._device)
        epoch = 0

        while self.iter < max_iter:

            epoch += 1

            for step in range(singe_batch_iter):
                self.iter += num_envs
                with torch.no_grad():
                    real_act_tuple = self.agent.predict_act(obs, is_continous_action=False, bin=cfg.bin)
                    if isinstance(real_act_tuple, Tuple):
                        real_act = real_act_tuple[0]
                        old_log_prob = real_act_tuple[1]
                        if len(real_act_tuple) == 3:
                            state_value = real_act_tuple[2]
                            state_value = state_value.view(-1)
                        else:
                            state_value = None
                    else:
                        real_act = real_act_tuple
                        old_log_prob = None

                    next_obs, rew, terminated, trunc, info = train_env.step(real_act)

                    if self._cfg.is_sparse_reward:
                        if isinstance(info['success'], torch.Tensor):
                            rew = info['success'].detach().to(device=self._device, dtype=torch.float32)
                        else:
                            rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

                        if 'final_info' in info.keys():
                            if isinstance(info['final_info']['success'], torch.Tensor):
                                rew += info['final_info']['success'].detach().to(device=self._device, dtype=torch.float32)
                            else:
                                rew += torch.tensor(info['final_info']['success'], device=self._device, dtype=torch.float32)

                        rew = rew.clamp(0, 1)  # success is 1, else 0
                        rew = rew - 1          # success is 0, else -1

                    rew = rew.view(-1)
                    done = torch.logical_or(terminated, trunc).to(dtype=torch.uint8)

                    rb.store(step, obs, rew, next_obs, done, real_act, old_log_prob, state_value)
            
                    obs = next_obs

            # rb.final_values = final_values
                
            print(" ---------------------- begin update {} ------------------".format(self.iter))
            metrics = self.agent.update_vector(rb, )
            print("the batch reward is {}, and success times is {}".format(metrics["batch_reward"], metrics["success_times"]))
            _to_log = []
            _to_log.append(metrics)
            _to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in _to_log]
            to_log += _to_log
            # rb.clear()

            should_test = self._cfg.evaluation.should and (epoch % self._cfg.evaluation.every == 0)
            if should_test:
                print("begin test")
                to_log += self.test_actor_critic(self._cfg.evaluation.eval_times)
                # ------------------------------------ wandb log ---------------------------------

            wandb_log(to_log, self.iter)
            to_log = []

    @torch.no_grad()
    def test_actor_critic(self, eval_times=25):
        test_env = self.test_env
        total_reward = 0.0
        success_rate = 0.0
        video_recorder = VideoRecorder(self._path_video_dir)
        for i in range(eval_times):
            seed = random.randint(0, 2**31 - 1)
            obs, _ = test_env.reset(seed=[seed + i for i in range(test_env.num_envs)])
            enabled=True if i==eval_times-1 else False
            video_recorder.init(obs, enabled=enabled)
            success = False
            steps = 0
            # hx, cx = hx.detach(), cx.detach()

            while True:
                act = self.agent.predict_act(obs, eval_mode=True)

                next_obs, rew, end, trunc, info = test_env.step(act)
                video_recorder.record(next_obs)
                # import pdb; pdb.set_trace()
                # video_recorder.save(f"{self.epoch}.mp4")
                steps += 1
                total_reward += rew.sum().item()

                # success |= bool(info['final_info']['success'])
                success |= bool(info['success'])
                if 'final_info' in info.keys():
                    if info['final_info']['success']:
                        success = True

                obs = next_obs

                if end or trunc or success:
                    break

            if success:
                success_rate += 1

            video_recorder.save(f"{self.iter}.mp4")
            # print("the {} traj success is {} and steps is {}".format(i, success, steps))


        success_rate = success_rate / eval_times
        avg_reward = total_reward / eval_times

        # 如果存在 set_old_policy 方法，且成功率大于最佳成功率，则更新最佳成功率
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.save_agents("best", self.best_success_rate)
            if hasattr(self.agent, "set_old_policy"):
                self.agent.set_old_policy()

        to_log = [{"success_rate": success_rate, "avg_reward": avg_reward}]

        print(f"{self._cfg.task} :Success rate: {success_rate}, Average reward: {avg_reward}")
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
            "agent": self.agent.state_dict(),
        }, agent_path)

    def save_eval_data(self, save_whole_traj=False,):

        dataset_name = f"data_/{self._cfg.task}_{self._cfg.img_size}/"
        # load policy
        self.bc_ckpt_dir = self._cfg.bc_ckpt_dir
        ckpt = torch.load(self.bc_ckpt_dir, self._device)
        self.agent.load_state_dict(ckpt["agent"])

        if save_whole_traj:
            partial_reset = False
        else:
            partial_reset = True

        train_env = self.registered_env_func(device=self._device, max_episode_steps=30, 
                                              partial_reset=partial_reset, num_envs=1, 
                                              size=self._cfg.img_size, 
                                              id=self._cfg.task,
                                              reconfiguration_freq=None,
                                              control_mode=self._cfg.control_mode)

        success_trajs = 10 ; success_traj = 0
        random_trajs = 10 ; random_traj = 0

        eps_imgs = []
        eps_actions = []
        eps_rewards = []
        eps_terminateds = []
        eps_truncateds = []
        eps_save_imgs_to_video = []

        success_data = []
        random_data = []

        video_record = VideoRecorder()

        seed = random.randint(0, 2**31 - 1)
        obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
        done = False
        eps_rew = 0
        step = 0

        success_flag = False

        while success_traj < success_trajs or random_traj < random_trajs:

            # if save_img:
                # 1 * 3 * 96 * 96 -> 96 * 96 * 3
            cam_img = obs.squeeze(0).add(1).div(2).mul(255).byte().permute(1, 2, 0)
            eps_save_imgs_to_video.append(cam_img.cpu().numpy())
            eps_imgs.append(cam_img)

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

                next_obs, rew, terminated, trunc, info = train_env.step(real_act)

            if self._cfg.is_sparse_reward:
                if isinstance(info['success'], torch.Tensor):
                    rew = info['success'].detach().to(device=self._device, dtype=torch.float32)
                else:
                    rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

                if 'final_info' in info.keys():
                    if isinstance(info['final_info']['success'], torch.Tensor):
                        rew += info['final_info']['success'].detach().to(device=self._device, dtype=torch.float32)
                    else:
                        rew += torch.tensor(info['final_info']['success'], device=self._device, dtype=torch.float32)

                rew = rew.clamp(0, 1)  # success is 1, else 0
                success_flag = rew.clone().bool()
                rew = rew - 1          # success is 0, else -1

            eps_actions.append(real_act)
            eps_rewards.append(rew)

            step += 1
            eps_rew += rew
                
            obs = next_obs

            eps_terminateds.append(terminated)
            eps_truncateds.append(trunc)
            
            ######## save data

            if save_whole_traj:
                flag = trunc
            else:
                flag = terminated or trunc or success_flag
            
            if flag:
                if success_flag and success_traj < success_trajs:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated, img_to_save) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds, eps_save_imgs_to_video
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            success_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })
                            if i == 0:
                                video_record.init(img_to_save)
                            else:
                                video_record.record(img_to_save)

                            path = f"{str(dataset_name)}/test_real_image/{int(success_traj)}/{int(i)}.png"
                        
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            cv2.imwrite(path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

                    success_traj += 1

                    video_path = f"{str(dataset_name)}/test_video/{success_traj-1}.mp4"
                    os.makedirs(os.path.dirname(f"{str(dataset_name)}/test_video/"), exist_ok=True)
                    video_record.save(video_path)

                    print("get success traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/test_raw_data"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/{success_traj-1}.npy"
                    np.save(data_path, success_data)
                
                elif not success_flag and random_traj < random_trajs:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated, img_to_save) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds, eps_save_imgs_to_video
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            random_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })
                            if i == 0:
                                video_record.init(img_to_save)
                            else:
                                video_record.record(img_to_save)

                            path = f"{str(dataset_name)}/test_real_image/random_{int(random_traj)}/{int(i)}.png"
                        
                            os.makedirs(os.path.dirname(path), exist_ok=True)
                            cv2.imwrite(path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))

                    random_traj += 1

                    video_path = f"{str(dataset_name)}/test_video/random_{random_traj-1}.mp4"
                    os.makedirs(os.path.dirname(f"{str(dataset_name)}/test_video/"), exist_ok=True)
                    video_record.save(video_path)

                    print("get random traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/test_raw_data"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/random_{random_traj-1}.npy"
                    np.save(data_path, random_data)

                else:
                    print("success traj is {} and random traj is {}".format(success_traj, random_traj))
                    print("get traj, the rew is {}".format(eps_rew))

                eps_imgs = []
                eps_actions = []
                eps_rewards = []
                eps_terminateds = []
                eps_truncateds = []
                eps_save_imgs_to_video = []
                seed = random.randint(0, 2**31 - 1)
                obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
                done = False
                success_flag = False
                eps_rew = 0
                step = 0
                random_data = []
                success_data = []


    def save_data(self, save_whole_traj=False,):

        assert save_whole_traj == False

        dataset_name = f"data_/{self._cfg.task}_{self._cfg.img_size}/"
        # load policy
        self.bc_ckpt_dir = self._cfg.bc_ckpt_dir
        ckpt = torch.load(self.bc_ckpt_dir, self._device)
        self.agent.load_state_dict(ckpt["agent"])

        if save_whole_traj:
            partial_reset = False
        else:
            partial_reset = True

        train_env = self.registered_env_func(device=self._device, max_episode_steps=30, 
                                              partial_reset=partial_reset, num_envs=1, 
                                              size=self._cfg.img_size, 
                                              id=self._cfg.task,
                                              reconfiguration_freq=None,
                                              control_mode=self._cfg.control_mode)

        success_trajs = 200 ; success_traj = 0         # 代表成功轨迹
        random_trajs = 200 ; random_traj = 0           # 代表失败轨迹

        eps_imgs = []
        eps_actions = []
        eps_rewards = []
        eps_terminateds = []
        eps_truncateds = []
        success_data = []
        random_data = []

        seed = random.randint(0, 2**31 - 1)
        obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
        done = False
        eps_rew = 0
        step = 0

        success_flag = False

        # total_traj = 150
        # cur_traj = 0

        while success_traj < success_trajs or random_traj < random_trajs:
        # while cur_traj < total_traj:

            cam_img = obs.squeeze(0).add(1).div(2).mul(255).byte().permute(1, 2, 0)
            eps_imgs.append(cam_img)

            with torch.no_grad():
                real_act_tuple = self.agent.predict_act(obs ) 
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

                next_obs, rew, terminated, trunc, info = train_env.step(real_act)

            if self._cfg.is_sparse_reward:
                if isinstance(info['success'], torch.Tensor):
                    rew = info['success'].detach().to(device=self._device, dtype=torch.float32)
                else:
                    rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

                if 'final_info' in info.keys():
                    if isinstance(info['final_info']['success'], torch.Tensor):
                        rew += info['final_info']['success'].detach().to(device=self._device, dtype=torch.float32)
                    else:
                        rew += torch.tensor(info['final_info']['success'], device=self._device, dtype=torch.float32)

                rew = rew.clamp(0, 1)  # success is 1, else 0
                success_flag = rew.clone().bool()
                rew = rew - 1          # success is 0, else -1

            eps_actions.append(real_act)
            eps_rewards.append(rew)

            step += 1
            eps_rew += rew

            eps_terminateds.append(terminated)
            eps_truncateds.append(trunc)

            # if self.domain_name == 'metaworld':
            #     # done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
            #     done = terminated or trunc
            #     success_flag |= bool(info['success'])
            #     eps_terminateds.append(terminated)
            #     eps_truncateds.append(trunc)
            # else:
            #     raise NotImplementedError("Only metaworld supported for now")\
                
            obs = next_obs
            
            ######## save data

            if save_whole_traj:
                flag = trunc
            else:
                flag = terminated or trunc or success_flag
            
            if flag:
                if success_flag and success_traj < success_trajs:
                # if success_flag:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds,
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            success_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })

                    success_traj += 1

                    print("get success traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/raw_data"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/{success_traj-1}.npy"
                    np.save(data_path, success_data)
                
                elif not success_flag and random_traj < random_trajs:
                # elif not success_flag:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated, ) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds, 
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            random_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })
                            
                    random_traj += 1

                    print("get random traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/raw_data"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/random_{random_traj-1}.npy"
                    np.save(data_path, random_data)

                else:
                    print("success traj is {} and random traj is {}".format(success_traj, random_traj))
                    print("get traj, the rew is {}".format(eps_rew))

                eps_imgs = []
                eps_actions = []
                eps_rewards = []
                eps_terminateds = []
                eps_truncateds = []
                # cur_traj += 1
                seed = random.randint(0, 2**31 - 1)
                obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
                success_flag = False
                eps_rew = 0
                step = 0
                random_data = []
                success_data = []
        
    def save_reward_data(self, np_seed=None, test_mode=False, save_whole_traj=False):

        if np_seed is not None:
            np.random.seed(np_seed)
            random.seed(np_seed)
            torch.manual_seed(np_seed)

        if save_whole_traj:
            partial_reset = False
        else:
            partial_reset = True

        train_env = self.registered_env_func(device=self._device, max_episode_steps=30, 
                                              partial_reset=partial_reset, num_envs=1, 
                                              size=self._cfg.img_size, 
                                              id=self._cfg.task,
                                              reconfiguration_freq=None,
                                              control_mode=self._cfg.control_mode)

        dataset_name = f"data_for_train_reward/{self._cfg.task}_{self._cfg.img_size}/"
        # load policy
        self.bc_ckpt_dir = self._cfg.bc_ckpt_dir
        ckpt = torch.load(self.bc_ckpt_dir, self._device)
        self.agent.load_state_dict(ckpt["agent"])

        if not test_mode:
            success_trajs = 250 ; success_traj = 0         # 代表成功轨迹
            random_trajs = 250 ; random_traj = 0           # 代表失败轨迹
            save_name = "raw_data_whole_traj" if save_whole_traj else "raw_data"
        else:
            success_trajs = 50 ; success_traj = 0
            random_trajs = 50 ; random_traj = 0
            save_name = "test_raw_data_whole_traj" if save_whole_traj else "test_raw_data"

        eps_imgs = []
        eps_actions = []
        eps_rewards = []
        eps_terminateds = []
        eps_truncateds = []
        success_data = []
        random_data = []

        seed = random.randint(0, 2**31 - 1)
        obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
        done = False
        eps_rew = 0
        step = 0

        success_flag = False

        # total_traj = 150
        # cur_traj = 0

        while success_traj < success_trajs or random_traj < random_trajs:
        # while cur_traj < total_traj:

            cam_img = obs.squeeze(0).add(1).div(2).mul(255).byte().permute(1, 2, 0)
            eps_imgs.append(cam_img)

            with torch.no_grad():
                real_act_tuple = self.agent.predict_act(obs ) 
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

                next_obs, rew, terminated, trunc, info = train_env.step(real_act)

            if self._cfg.is_sparse_reward:
                if isinstance(info['success'], torch.Tensor):
                    rew = info['success'].detach().to(device=self._device, dtype=torch.float32)
                else:
                    rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

                if 'final_info' in info.keys():
                    if isinstance(info['final_info']['success'], torch.Tensor):
                        rew += info['final_info']['success'].detach().to(device=self._device, dtype=torch.float32)
                    else:
                        rew += torch.tensor(info['final_info']['success'], device=self._device, dtype=torch.float32)

                rew = rew.clamp(0, 1)  # success is 1, else 0
                success_flag = rew.clone().bool()

            eps_actions.append(real_act)
            eps_rewards.append(rew)

            step += 1
            eps_rew += rew

            eps_terminateds.append(terminated)
            eps_truncateds.append(trunc)

            # if self.domain_name == 'metaworld':
            #     # done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
            #     done = terminated or trunc
            #     success_flag |= bool(info['success'])
            #     eps_terminateds.append(terminated)
            #     eps_truncateds.append(trunc)
            # else:
            #     raise NotImplementedError("Only metaworld supported for now")\
                
            obs = next_obs
            
            ######## save data

            if save_whole_traj:
                flag = trunc
            else:
                flag = terminated or trunc or success_flag
            
            if flag:
                if success_flag and success_traj < success_trajs:
                # if success_flag:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds,
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            success_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })

                    success_traj += 1

                    print("get success traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/{save_name}"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/{success_traj-1}.npy"
                    np.save(data_path, success_data)
                
                elif not success_flag and random_traj < random_trajs:
                # elif not success_flag:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds)
                    for i, (img, action, reward, terminated, truncated, ) in enumerate(zip(
                        eps_imgs, eps_actions, eps_rewards, eps_terminateds, eps_truncateds, 
                        )):
                            img, action, reward, terminated, truncated = to_np((img, action, reward, terminated, truncated))
                            random_data.append({
                            'image': img,
                            'action': action,
                            'reward': reward,
                            'terminated': terminated,
                            'truncated': truncated,
                        })
                            
                    random_traj += 1

                    print("get random traj, the rew is {}".format(eps_rew))

                    data_dir = f"{str(dataset_name)}/{save_name}"
                    os.makedirs(data_dir, exist_ok=True)
                    data_path = f"{data_dir}/random_{random_traj-1}.npy"
                    np.save(data_path, random_data)

                else:
                    print("success traj is {} and random traj is {}".format(success_traj, random_traj))
                    print("get traj, the rew is {}".format(eps_rew))

                eps_imgs = []
                eps_actions = []
                eps_rewards = []
                eps_terminateds = []
                eps_truncateds = []
                # cur_traj += 1
                seed = random.randint(0, 2**31 - 1)
                obs, _ = train_env.reset(seed=[seed + i for i in range(train_env.num_envs)])
                success_flag = False
                eps_rew = 0
                step = 0
                random_data = []
                success_data = []