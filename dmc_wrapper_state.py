from __future__ import annotations
import gymnasium as gym
import dmc2gym
import numpy as np
import torch
from gymnasium.spaces import Box
from typing import Any, Dict, Tuple

class DMCWrapper(gym.Wrapper):
    def __init__(self, domain_name: str, task_name: str, 
                 seed: int = 0, 
                 visualize_reward: bool = False, 
                 from_pixels: bool = False,
                 height: int = 84, width: int = 84,
                 frame_skip: int = 1, 
                 device: torch.device = torch.device("cuda")):
        
        env = dmc2gym.make(
            domain_name=domain_name, 
            task_name=task_name, 
            seed=seed,
            visualize_reward=visualize_reward,
            from_pixels=from_pixels,
            height=height, width=width,
            frame_skip=frame_skip
        )
        super().__init__(env)

        self.device = device
        self.from_pixels = from_pixels
        self.frame_skip = frame_skip

        if from_pixels:
            self.observation_space = Box(low=0, high=255, shape=(3, height, width), dtype=np.uint8)
        else:
            self.observation_space = env.observation_space
        
        self.action_space = env.action_space
    
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_tensor(obs), info

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action.cpu().numpy())
        obs, reward, done, truncated = (self._to_tensor(x) for x in (obs, reward, done, truncated))
        return obs, reward, done, truncated, info

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        return self.env.render(mode=mode)

    def _to_tensor(self, x: Any) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, (float, int, bool)):
            return torch.tensor([x], dtype=torch.float32, device=self.device)
        return x
