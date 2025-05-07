from __future__ import annotations
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from omegaconf import ListConfig
import numpy as np
import torch
from torch import Tensor
import mujoco
# from .atari_preprocessing import AtariPreprocessing
import sys
import os

from gymnasium.spaces import Box
from gymnasium.core import WrapperActType, WrapperObsType
import os
import cv2
os.environ["MUJOCO_GL"] = "egl"


def make_mw_env(
    id: str,
    num_envs: int,
    device: torch.device,
    size: int,
    max_episode_steps: Optional[int] = 100,
    frame_skip: int = 2,
    is_sparse_reward: bool = False,
    frame_stack: int = 1,
) -> TorchEnv:
    
    if isinstance(size, ListConfig):
        assert len(size) == 2  # H * W
        size = size[0]

    def env_fn():
        env = MetaWorldEnv(id, frame_skip, device, size, max_episode_steps, is_sparse_reward,
                           frame_stack=frame_stack)
        return env

    # TODO: AsyncVectorEnv IN metaworld ?
    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
    # env = env_fn()

    env = TorchEnv(env, device)

    return env

class MetaWorldEnv(gymnasium.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task, frame_skip: int, device="cuda:0", 
                 screen_size=128, max_episode_steps=100,
                 is_sparse_reward=False, frame_stack=1):
        super(MetaWorldEnv, self).__init__()

        # init params
        self.is_sparse_reward = is_sparse_reward
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack

        from metaworld.envs import (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
        )
        task = f"{task}-goal-observable"

        # set env params
        self.env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]()
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # clip camera position
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        # self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        # set render size
        self.env.mujoco_renderer.width = screen_size
        self.env.mujoco_renderer.height = screen_size
        self.env.model.vis.global_.offwidth = screen_size
        self.env.model.vis.global_.offheight = screen_size
        # set render mode
        self.env.render_mode = "rgb_array"
        # set render camera
        self.env.mujoco_renderer.camera_id = mujoco.mj_name2id(
            self.env.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "corner2",
        )
        
        # self.env.model.vis.map.znear = 0.1
        # self.env.model.vis.map.zfar = 1.5

        def get_last_device_id(device: torch.device) -> int:
            if device.type == "cuda" and device.index is not None:
                return device.index
            return -1  # 如果 device 不是 CUDA 设备或没有索引，返回 -1 表示无效
        
        self.device_id = get_last_device_id(device)
        
        self.max_episode_steps = max_episode_steps
        self.screen_size = screen_size  
        
        _low, _high, _obs_dtype = (0, 255, np.uint8)
        _shape = (screen_size, screen_size, 3 * frame_stack)
        self.observation_space = Box(low=_low, high=_high, shape=_shape, dtype=_obs_dtype)
        self.action_space = Box(low=-1., high=1., shape=(4,), dtype=np.float32)

        self._obs_buffer = deque([], maxlen=frame_stack)

        # init
        self.cur_step = 0

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        self.env.camera_name="corner2"
        img = self.env.render()
        cam_img = np.flipud(img).copy()
        return cam_img

    def step(self, action: np.array):

        total_reward, terminated, truncated, info = 0.0, False, False, {}
    
        for t in range(self.frame_skip):
            self.cur_step += 1            
            state, reward, terminated, truncated, info = self.env.step(action)
            # if self.is_sparse_reward:
            #     reward = 1.0 if info["success"] else 0.0
            total_reward += reward
            if self.is_sparse_reward:
                total_reward = 1.0 if info["success"] else 0.0
            self.game_over = terminated
            # if self.ale.lives() < self.lives:
            #     life_loss = True
            #     self.lives = self.ale.lives()

            if terminated or truncated:
                self.env.reset()
                break

            # if t == self.frame_skip - 2:
            #     # cam_img = self.env.render()
            #     # self.obs_buffer[1][i] = np.flipud(cam_img).copy()
            #     self.obs_buffer[1] = self.get_rgb()
            # elif t == self.frame_skip - 1:
            #     # cam_img = self.env.render()
            #     # self.obs_buffer[0][i] = np.flipud(cam_img).copy()
            #     self.obs_buffer[0] = self.get_rgb()
        obs = self.get_rgb()
        self._obs_buffer.append(obs)

        if self.cur_step >= self.max_episode_steps:
            truncated = 1.0

        stack_obs = self._get_obs()
        # info["original_obs"] = original_obs
        # truncated remains False all the time

        return stack_obs, total_reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Resets the environment using preprocessing."""
        # NoopReset
        reset_info = {}

        state, reset_info = self.env.reset(seed=seed, options=options)
        cam_img = self.get_rgb() # get the first observation

        for _ in range(self.frame_stack):
            self._obs_buffer.append(cam_img)

        stack_obs = self._get_obs()
        reset_info["success"] = 0
        self.cur_step = 0
        # print("executing metaworld env reset")

        # obs_dict = {"obs": obs, "state": state}

        # return obs_dict, reset_info
        return stack_obs, reset_info  

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    # def render(self, mode='rgb_array'):
    #     img = self.get_rgb()
    #     return img

    def _get_obs(self):
        obs = np.concatenate(list(self._obs_buffer), axis=-1)     #  84 * 84 * (3 * frame_stack)
        return obs

class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.env = env
        self.device = device
        if len(env.observation_space.shape) == 3:
            raise ValueError("The observation space should have 4 dimensions, first dim is the number of envs")
            self.num_envs = 1
            b = 1
            h, w, c = env.observation_space.shape
        elif len(env.observation_space.shape) == 4:
            self.num_envs = env.observation_space.shape[0]
            b, h, w, c = env.observation_space.shape

        self.num_actions = env.action_space.shape[1]
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, self.num_actions))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)

        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        # actions: B x Da (B is the number of envs, Da = 4 in the case of metaworld)
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())

        dead = np.logical_or(end, trunc)
        if dead.any():
            info["final_observation"] = deepcopy(obs)
            info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        # import pdb; pdb.set_trace()
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

# test
if __name__ == "__main__":

    device = torch.device("cuda:1")
    mw_env = make_mw_env("button-press-topdown-v2", 1, device, 128, frame_stack=3)
    obs, info = mw_env.reset()
    while(1):
        action = torch.rand((1, 4))
        obs, rew, end, trunc, loop_info = mw_env.step(action)
        if end.any() or trunc.any():
            print("end")
            break
    print("info", loop_info['success'])
    print("end", end)
    print("trunc", trunc)
    print("obs", obs.shape)

    # print(obs.shape)
    # obs_np = obs[ :, :, -3:].add(1).div(2).mul(255).byte().permute(2,0,1).cpu().numpy()
    obs_np = obs[0, -3:, :, :].add(1).div(2).mul(255).byte().permute(1,2,0).cpu().numpy()
    import matplotlib.pyplot as plt
    plt.imshow(obs_np, cmap='viridis')  # Adjust cmap based on your data
    plt.colorbar()
    plt.title("Observation")
    plt.savefig("observation.png")