from collections import deque
from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
from dm_env import specs
from dm_control import suite
from dm_control.suite.wrappers import pixels
import gymnasium
import torch
from torch import Tensor
from typing import Any, Dict, Tuple

import os
os.environ["MUJOCO_GL"] = "egl"

class DMCGymWrapper(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, domain, task, seed=42, img_size=84, camera_id=0, frame_skip=2, frame_stack=1):
        self._env = suite.load(domain, task, task_kwargs={'random': seed})
        self.frame_skip = frame_skip
        self.seed_value = seed

        # Wrap for pixel observations
        render_kwargs = dict(height=img_size, width=img_size, camera_id=camera_id)
        self._env = pixels.Wrapper(self._env, pixels_only=True, render_kwargs=render_kwargs)

        # Observation sapce
        obs_spec: specs.Array = self._env.observation_spec()['pixels']
        h, w, c = obs_spec.shape
        c = c * frame_stack
        self.observation_space = Box(low=0, high=255, shape=(h,w,c), dtype=np.uint8)

        # action space
        act_spec = self._env.action_spec()
        self.action_space = Box(low=act_spec.minimum, high=act_spec.maximum, shape=act_spec.shape, dtype=np.float32)

        self.obs_buffer = deque([], maxlen=frame_stack)

    def reset(self, seed=None, options=None):
        timestep = self._env.reset()
        obs = timestep.observation['pixels']

        for i in range(self.obs_buffer.maxlen):
            self.obs_buffer.append(obs)

        stack_obs = np.concatenate(list(self.obs_buffer), axis=-1)

        return stack_obs, {}

    def step(self, action):
        total_reward, done = 0.0, False
        for _ in range(self.frame_skip):
            timestep = self._env.step(action)
            total_reward += timestep.reward
            done = timestep.last()
            if done:
                break
        obs = timestep.observation['pixels']
        self.obs_buffer.append(obs)

        stack_obs = np.concatenate(list(self.obs_buffer), axis=-1)

        return stack_obs, total_reward, done, False, {}

    def render(self):
        return self._env.physics.render(height=84, width=84, camera_id=0)
    
class TorchEnv(Env):
    def __init__(self, env: gymnasium.Env, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.env = env
        self.device = device

        # observation space
        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w), dtype=np.float32)

        # action space
        _, self.num_actions = env.action_space.shape 
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, self.num_actions,), dtype=np.float32)

        #
        self.num_envs = b

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = self._to_tensor(obs)
        return obs, info

    def step(self, action: torch.Tensor):
        obs, reward, terminate, trunc, info = self.env.step(action.cpu().numpy())
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, reward, terminate, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray) and x.ndim == 3:
            raise NotImplementedError
        elif isinstance(x, np.ndarray) and x.ndim == 4:
            # batch obs (B, H, W, C)
            return torch.tensor(x, dtype=torch.float32, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        
    def render(self):
        return self.env.render()

from gymnasium.vector import SyncVectorEnv

def get_dmc_env(id: str, frame_skip=2, num_envs=1, size=84, 
                frame_stack=3, device=torch.device("cuda")):
    domain, task = id.split('_', 1)

    def make_env_fn():
        return DMCGymWrapper(domain, task, frame_skip=frame_skip, img_size=size, frame_stack=frame_stack)

    env_fns = [make_env_fn for _ in range(num_envs)]
    vec_env = SyncVectorEnv(env_fns)
    torch_env = TorchEnv(vec_env, device)
    return torch_env

if __name__ == "__main__":
    env = get_dmc_env("walker_walk", frame_stack=3)
    obs, info = env.reset()
    print(obs.shape)
    print(obs.max(), obs.min(), obs.mean())
    action = torch.tensor(env.action_space.sample())
    obs, reward, done, _, info = env.step(action)
    print(obs.shape, reward, done)
    env.render()
