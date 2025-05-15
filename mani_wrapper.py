from copy import deepcopy
from typing import Any, Dict, Tuple
import gymnasium as gym
import numpy as np
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import torch
from torch import Tensor

def make_mani_env(id, device, size, num_envs, 
                  partial_reset=True,
                  reconfiguration_freq=None,
                  max_episode_steps=50,):
    # default args
    reconfiguration_freq = None

    env_kwargs = dict(obs_mode="rgb", render_mode="rgb_array", 
                      sim_backend="physx_cuda", 
                      control_mode="pd_joint_delta_pos", reward_mode='sparse',)
    # eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(id, num_envs=num_envs, 
                    reconfiguration_freq=reconfiguration_freq, 
                    max_episode_steps=max_episode_steps,
                    **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=False)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=not partial_reset, record_metrics=True, )
    envs = TorchEnv(envs, device,)
    return envs

class TorchEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, device: torch.device,) -> None:
        super().__init__(env)
        self.env = env
        self.device = device


        if len(env.observation_space['rgb'].shape) == 3:
            raise ValueError("obs shape should be 4 dim")
            b = 1
            h, w, c = env.observation_space.shape
        elif len(env.observation_space['rgb'].shape) == 4:
            self.num_envs = env.observation_space['rgb'].shape[0]
            b, h, w, c = env.observation_space['rgb'].shape

        self.num_actions = env.action_space.shape[1]
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(b, c, h, w))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(b, self.num_actions))

    def reset(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        obs_dict, info = self.env.reset(*args, **kwargs)
        obs = obs_dict['rgb']

        return self._to_tensor(obs), info

    def step(self, actions: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        # actions: B x Da (B is the number of envs, Da = 4 in the case of metaworld)
        obs_dict, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        obs = obs_dict['rgb']

        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        
        if "final_info" in info:
            final_obs = info["final_observation"]['rgb']
            info["final_observation"] = self._to_tensor(final_obs)
        
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--render_mode", type=str, default="all")
    args = parser.parse_args()

    env = make_mani_env(args.env_id, args.num_envs, render_mode=args.render_mode)
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        action = torch.from_numpy(action)
        obs, reward, terminate, trunc, info = env.step(action)
        if terminate or trunc:
            obs, info = env.reset()
    import pdb; pdb.set_trace()

