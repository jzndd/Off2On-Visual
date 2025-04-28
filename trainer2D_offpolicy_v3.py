from collections import deque
import os
from pathlib import Path
import random
from typing import List, Tuple, Union

import cv2
import torch
from omegaconf import DictConfig, OmegaConf
from rl_agent import PPOAgent2D, VRL3Agent, DrQv2Agent
import wandb
import numpy as np
from rl_agent.utils import EfficientReplayBuffer 

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
        self._cfg = cfg
        # self._path_video_dir = os.path.join(root_dir, "videos")
        # os.makedirs(self._path_video_dir, exist_ok=True)
        self._path_video_dir = Path("videos")
        os.makedirs(self._path_video_dir, exist_ok=True)
        self._save_path = Path("saved_models")
        self._save_path.mkdir(parents=True, exist_ok=True)

        if cfg.task != "walker_walk":
            from mw_wrapper import make_mw_env
            self.registered_env_func = make_mw_env
            self.domain_name = 'metaworld'
        else:
            from dmc_wrapper import get_dmc_env
            self.registered_env_func = get_dmc_env
            self.domain_name = 'dmc'

        env = self.registered_env_func(num_envs=1,  device=self._device, **cfg.env.train)
        cfg.agent.actor_critic_cfg.num_actions = deepcopy(env.num_actions)
        # cfg.agent.actor_critic_cfg.num_states = deepcopy(env.num_states)

        # self.test_env = self.registered_env_func(num_envs=1, device=self._device, **cfg.env.test)
        # obs, info = self.test_env.reset()
        # init agent
        self.agent: DrQv2Agent = DrQv2Agent(cfg.agent.actor_critic_cfg, cfg.agent.drqv2_cfg)
        self.agent.to(self._device)
        self.agent.setup_training(cfg.actor_critic.actor_critic_loss)

        self.metrics = -0.01

        # print(os.path.exists(cfg.bc_ckpt_dir))
        # print(cfg.bc_ckpt_dir)
        # exit(1)

        self.bc_ckpt_dir = cfg.bc_ckpt_dir

        self.best_success_rate = -0.01

    def run(self):
        cfg = self._cfg
        train_env = self.registered_env_func(num_envs=cfg.collection.train.num_envs, device=self._device, **cfg.env.train)
    
        max_iter = self._cfg.training.online_max_iter

        # assert self._cfg.actor_critic.training.batch_size >= 1024  # PPO's batch_size >= 1024
        # rb = OnlineReplayBuffer(1000000, train_env.observation_space.shape[1:], (train_env.action_space.shape[1],))

        to_log = []

        if self._cfg.expert_rb_dir is not None and self._cfg.train_with_offline_data_mode in ['rlpd', 'vrl3', 'drqv2offline']:
            if self.domain_name == 'metaworld':
                if self._cfg.is_sparse_reward:
                    self._cfg.expert_rb_dir = self._cfg.expert_rb_dir.replace("reward", "reward_sparse")
                if self._cfg.is_whole_traj:
                    self._cfg.expert_rb_dir = self._cfg.expert_rb_dir.replace(".pkl", "_whole_traj_50traj.pkl")

            elif self.domain_name == 'dmc':
                if self._cfg.frame_stack > 1:
                    self._cfg.expert_rb_dir = self._cfg.expert_rb_dir.replace(".pkl", f"_stack{self._cfg.frame_stack}.pkl")
            
            expertrb = EfficientReplayBuffer(1000000, train_env.observation_space.shape[1:], 
                                           (train_env.action_space.shape[1],),
                                           frame_stack=self._cfg.frame_stack)
            expertrb.load(self._cfg.expert_rb_dir)

        else:
            expertrb = None

        # First stage: BC
        if self._cfg.train_with_bc:
            
            if os.path.exists(self.bc_ckpt_dir):
                ckpt = torch.load(self.bc_ckpt_dir, self._device)
                # ckpt = torch.load(self.bc_ckpt_dir, self._device, weights_only=True)
                self.agent.load_state_dict(ckpt["agent"])
                print("load bc ckpt from {}".format(self.bc_ckpt_dir))
            else:
                for i in range(cfg.training.offline_steps):
                    metrics = self.agent.update(None, i, expertrb)
                    if i % 1000 == 0:
                        print(f"bc_actor_warmup_steps: {i}")
                        print(metrics)
                    _to_log = []
                    _to_log.append(metrics)
                    _to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in _to_log]
                    to_log += _to_log
                    if i % 5000 == 0:
                        _to_log = self.test_actor_critic(eval_times=10)
                        to_log += _to_log

                    wandb_log(to_log, i)
                    to_log = []

                os.makedirs(os.path.dirname(self.bc_ckpt_dir), exist_ok=True)
                torch.save({
                    "agent": self.agent.state_dict(),
                }, self.bc_ckpt_dir)

        self.agent.bc_transfer_ac()

        # use RLPD, a empty buffer
        rb = EfficientReplayBuffer(1000000, train_env.observation_space.shape[1:], 
                                 (train_env.action_space.shape[1],),
                                 frame_stack=self._cfg.frame_stack)
        
        if self._cfg.train_with_offline_data_mode == 'vrl3':
            rb = deepcopy(expertrb)

        # obs_buffer = deque([], maxlen=cfg.frame_stack)
        if self._cfg.only_bc:
            exit(1) 

        while self.iter < max_iter:

            # -----------------------------   2-stage params update   -----------------------------
            obs, _ = train_env.reset()
            rb.store(obs.detach().cpu().numpy(), None, None, None, True)

            done = False
            eps_rew = 0
            step = 0

            # 2 stage : ac update
            # end_traj_flag = False
            # while not done:           
            while not done:
                self.iter += 1
                with torch.no_grad():
                    real_act_tuple = self.agent.predict_act(obs, step=self.iter)
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

                    obs, rew, terminated, trunc, info = train_env.step(real_act)

                    if self.domain_name == "metaworld" and self._cfg.is_sparse_reward:
                        rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

                    step += 1
                    eps_rew += rew

                    if self.domain_name == 'metaworld':
                        # done here means win
                        # terminated here is always False
                        # trunc here means time limit
                        if not self._cfg.is_whole_traj:
                            done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
                        else:
                            done = False
                        done = done or terminated or trunc
                        dw = torch.tensor(done.bool() & (~trunc.bool()), dtype=torch.uint8, device=done.device)
                    else:
                        done = terminated
                        # rerference: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/8f767b99ad44990b49f6acf3159660c5594db77e/5.PPO-continuous/PPO_continuous_main.py#L100
                        dw = torch.tensor(done.bool() & (~trunc.bool()), dtype=torch.uint8, device=done.device) # loss and win but not trunc (because if loss, there is no next state)

                    # rb.store(obs, next_obs, rew, done, real_act, old_log_prob=old_log_prob, 
                    #         state_value=state_value, dw=dw)
                    # rb.store(obs.detach().cpu().numpy(), real_act.detach().cpu().numpy(), next_obs.detach().cpu().numpy(), 
                    #          rew.detach().cpu().numpy(), done.detach().cpu().numpy())
                    discount = 1 - done
                    # obs = next_obs

                    rb.store(obs.detach().cpu().numpy(), real_act.detach().cpu().numpy(),
                             rew.detach().cpu().numpy(), discount.detach().cpu().numpy(), False)

                # ------------------------------------ should train ? ---------------------------------
                
                # if rb.size >= :
                    # print(" ---------------------- begin update {} ------------------".format(self.iter))
                # always train
                if self.iter >= 4000:
                    metrics = self.agent.update(rb, self.iter, expertrb)
                    _to_log = []
                    _to_log.append(metrics)
                    _to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in _to_log]
                    to_log += _to_log

                # ------------------------------------ should test ? ---------------------------------

                should_test = self._cfg.evaluation.should and (self.iter % self._cfg.evaluation.every_iter == 0)
                if should_test:
                    print("begin test")
                    to_log += self.test_actor_critic(self._cfg.evaluation.eval_times)

                # ------------------------------------ wandb log ---------------------------------

                wandb_log(to_log, self.iter)
                to_log = []

            print("this traj eps rew is {}, and the traj len is {}".format(eps_rew, step))

    @torch.no_grad()
    def test_actor_critic(self, eval_times=25):
        test_env = self.registered_env_func(num_envs=self._cfg.collection.test.num_envs, device=self._device, **self._cfg.env.test)
        total_reward = 0.0
        success_rate = 0.0
        video_recorder = VideoRecorder(self._path_video_dir)
        for i in range(eval_times):
            seed = random.randint(0, 2**31 - 1)
            obs, _ = test_env.reset(seed=[seed + i for i in range(test_env.num_envs)])
            # enabled=True if i==eval_times-1 else False
            enabled=False
            video_recorder.init(obs, enabled=enabled)
            success = False
            steps = 0
            # hx, cx = hx.detach(), cx.detach()

            while True:
                
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

                if self.domain_name == 'metaworld':

                    success |= bool(info['success'])

                obs = next_obs

                if end or trunc:
                    break

            if success:
                success_rate += 1

            video_recorder.save(f"{self.iter}.mp4")
            print("the {} traj success is {}, steps is {}".format(i, success, steps))


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
        # save_frames = np.array(video_recorder.frames).transpose(0, 3, 1, 2)
        # video_wandb = wandb.Video(save_frames, fps=video_recorder.fps, format="mp4")
        # to_log.append({"video": video_wandb})

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

    def save_eval_data(self, ):

        dataset_name = f"data_/{self._cfg.task}_{self._cfg.img_size}/"
        # load policy
        self.bc_ckpt_dir = self._cfg.bc_ckpt_dir
        ckpt = torch.load(self.bc_ckpt_dir, self._device)
        self.agent.load_state_dict(ckpt["agent"])

        train_env = self.registered_env_func(num_envs=1, device=self._device, **self._cfg.env.train)

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
                rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

            eps_actions.append(real_act)
            eps_rewards.append(rew)

            step += 1
            eps_rew += rew

            if self.domain_name == 'metaworld':
                # done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
                done = terminated or trunc
                success_flag |= bool(info['success'])
                eps_terminateds.append(terminated)
                eps_truncateds.append(trunc)
            else:
                raise NotImplementedError("Only metaworld supported for now")\
                
            obs = next_obs
            
            ######## save data
            
            if trunc:
                if success_flag and success_traj < success_trajs:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds) == 50
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
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds) == 50
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


    def save_data(self,):

        dataset_name = f"data_/{self._cfg.task}_{self._cfg.img_size}/"
        # load policy
        self.bc_ckpt_dir = self._cfg.bc_ckpt_dir
        ckpt = torch.load(self.bc_ckpt_dir, self._device)
        self.agent.load_state_dict(ckpt["agent"])

        train_env = self.registered_env_func(num_envs=1, device=self._device, **self._cfg.env.train)

        success_trajs = 75 ; success_traj = 0         # 代表成功轨迹
        random_trajs = 75 ; random_traj = 0           # 代表失败轨迹

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
                rew = torch.tensor(info['success'], device=self._device, dtype=torch.float32)

            eps_actions.append(real_act)
            eps_rewards.append(rew)

            step += 1
            eps_rew += rew

            if self.domain_name == 'metaworld':
                # done = torch.tensor(info['success'], device=self._device, dtype=torch.float32)
                done = terminated or trunc
                success_flag |= bool(info['success'])
                eps_terminateds.append(terminated)
                eps_truncateds.append(trunc)
            else:
                raise NotImplementedError("Only metaworld supported for now")\
                
            obs = next_obs
            
            ######## save data
            
            if trunc:
                if success_flag and success_traj < success_trajs:
                # if success_flag:
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds) == 50
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
                    assert len(eps_imgs) == len(eps_actions) == len(eps_rewards) == len(eps_terminateds) == len(eps_truncateds) == 50
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
        