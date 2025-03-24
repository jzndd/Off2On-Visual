import gym
from omegaconf import ListConfig
import d4rl
import numpy as np
import torch
from gymnasium.spaces import Box
from typing import Any, Dict, Tuple
import gymnasium
from gymnasium.vector import AsyncVectorEnv

class D4RLWrapper(gymnasium.Wrapper):
    def __init__(self, env_name: str, frame_skip: int = 2, max_episode_steps=None):
        env = gym.make(env_name)
        super().__init__(env)
        
        self.env.render_mode = 'rgb_array'
        self.frame_skip = frame_skip
        self.observation_space = Box(low=-1, high=1, shape=env.observation_space.shape)
        self.action_space = Box(low=-1, high=1, shape=(env.action_space.shape[0],))
        
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else env._max_episode_steps
        self.cur_step = 0
    
    # def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    def reset(self, seed = None, options = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fix: Ensure reset returns (obs, info)"""
        obs = self.env.reset(seed=seed, options=options)
        self.cur_step = 0
        if isinstance(obs, tuple):  # If already a tuple (gymnasium format), return as is
            return obs
        return obs, {}  # Convert to tuple if missing `info`

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        total_reward = 0.0
        done, truncated = False, False
        
        for _ in range(self.frame_skip):
            info_tuple = self.env.step(action)
            obs, reward, done, info = info_tuple
            self.cur_step += 1
            # if len(info_tuple) == 4:
            #     obs, reward, done, info = info_tuple
            #     truncated = False
            # else:
            #     obs, reward, done, truncated, info = info_tuple
            if self.cur_step == self._max_episode_steps:
                truncated = True

            total_reward += reward
            if done or truncated:
                break
        
        # obs, reward, done, truncated = (self._to_tensor(x) for x in (obs, total_reward, done, truncated))
        return obs, reward, done, truncated, info
    
    def render(self, render_mode: str = 'rgb_array') -> np.ndarray:
        return self.render()
    
    def get_normalized_score(self, score: float) -> float:
        return self.env.get_normalized_score(score)
    
    # def _to_tensor(self, x: Any) -> torch.Tensor:
    #     if isinstance(x, np.ndarray):
    #         return torch.tensor(x, dtype=torch.float32, device=self.device)
    #     elif isinstance(x, (float, int, bool)):
    #         return torch.tensor([x], dtype=torch.float32, device=self.device)
    #     return x

class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.env = env
        self.device = device

        if len(env.observation_space.shape) == 2:
            self.num_envs = env.observation_space.shape[0]
            b, state_dim = env.observation_space.shape
        else:
            raise ValueError("The observation space should have 2 or 4 dimensions")

        self.num_actions = env.action_space.shape[1]
        self.num_states = env.observation_space.shape[1]

        # if len(env.observation_space.shape) == 4:
        #     self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))
        # elif len(env.observation_space.shape) == 2:
        self.observation_space = gymnasium.spaces.Box(low=-torch.inf, high=torch.inf, shape=(b, state_dim))
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, self.num_actions))

    def reset(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)

        return self._to_tensor(obs), info

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # actions: B x Da (B is the number of envs, Da = 4 in the case of metaworld)
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())

        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        
    def get_normalized_score(self, score: float) -> float:
        return self.env.env_fns[0]().get_normalized_score(score) * 100

def get_d4rl_env(id: str, frame_skip: int = 1, device: torch.device = torch.device("cuda"), num_envs=1, size=(84,), max_episode_steps=None) -> D4RLWrapper:
    if isinstance(size, ListConfig):
        assert len(size) == 2  # H * W
        size = size[0]

    def env_fn():
        env = D4RLWrapper(env_name=id, frame_skip=frame_skip)
        return env

    # TODO: AsyncVectorEnv IN metaworld ?
    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    env = TorchEnv(env, device)

    return env

if __name__ == "__main__":
    env = get_d4rl_env("walker2d-medium-v2")
    obs, info = env.reset()
    print(obs)
    print(env.observation_space)
    print(env.action_space)

    actions = torch.rand((1, 6))
    obs, rew, end, trunc, info = env.step(actions)
    print(obs)

    score = env.get_normalized_score(1000)
    print(score)
    # print(env.step(env.action_space.sample()))
    # print(env.render())
