import gym
from omegaconf import ListConfig
import d4rl
import numpy as np
import torch
from gymnasium.spaces import Box
from typing import Any, Dict, Tuple
import gymnasium
from gymnasium.vector import AsyncVectorEnv

from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels

class D4RLWrapper(gymnasium.Wrapper):
    def __init__(self, env, frame_skip: int = 2, max_episode_steps=None, seed=42):
        # env = gym.make(env_name)
        super().__init__(env)
        
        self.env.render_mode = 'rgb_array'
        self.frame_skip = frame_skip
        import pdb; pdb.set_trace()
        self.observation_space = Box(low=0, high=255, shape=env.observation_spec()['pixels'].shape)
        self.action_space = Box(low=-1, high=1, shape=env.action_spec().shape)
        
        # self._max_episode_steps = max_episode_steps if max_episode_steps is not None else env._max_episode_steps
        self.cur_step = 0
    
    # def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    def reset(self, seed = None, options = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fix: Ensure reset returns (obs, info)"""
        timestep = self.env.reset(seed=seed, options=options)
        import pdb; pdb.set_trace()
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

            total_reward += reward
            if done or truncated:
                break
        
        # obs, reward, done, truncated = (self._to_tensor(x) for x in (obs, total_reward, done, truncated))
        return obs, reward, done, truncated, info
    
    def render(self, render_mode: str = 'rgb_array') -> np.ndarray:
        return self.render()

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
        import pdb; pdb.set_trace()
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

def get_d4rl_env(id: str, frame_skip: int = 1, device: torch.device = torch.device("cuda"), num_envs=1, size=(84,), max_episode_steps=None) -> D4RLWrapper:
    
    if isinstance(size, ListConfig):
        assert len(size) == 2  # H * W
        size = size[0]

    def make_base_pixel_env(id, seed: int = 42):
        pixel_hw = size
        if 'offline' in id:
            id = '_'.join(id.split('_')[1:3])
        domain, task = id.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)

        if (domain, task) in suite.ALL_TASKS:
            env = suite.load(domain,
                             task,
                             task_kwargs={'random': seed},
                             visualize_reward=False)
            pixels_key = 'pixels'
        else:
            name = f'{domain}_{task}_vision'
            env = manipulation.load(name, seed=seed)
            pixels_key = 'front_close'

        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=pixel_hw, width=pixel_hw, camera_id=camera_id)
        env = pixels.Wrapper(env,
                                 pixels_only=True,
                                 render_kwargs=render_kwargs)
            
        return env

    def env_fn():
        env = make_base_pixel_env(id)
        env = D4RLWrapper(env, frame_skip=frame_skip, max_episode_steps=max_episode_steps)
        # env = D4RLWrapper(env_name=id, frame_skip=frame_skip)
        return env

    # TODO: AsyncVectorEnv IN metaworld ?
    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    env = TorchEnv(env, device)

    return env

if __name__ == "__main__":
    env = get_d4rl_env("walker_walk")
    timestep = env.reset()
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
