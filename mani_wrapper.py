import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def make_mani_env(id, num_envs, render_mode="rgb_array", sim_backend="physx_cuda", control_mode="pd_joint_delta_pos"):
    # default args
    eval_reconfiguration_freq = 1
    reconfiguration_freq = None

    env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda", control_mode="pd_joint_delta_pos")
    # eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(id, num_envs=num_envs, reconfiguration_freq=reconfiguration_freq, **env_kwargs)
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=False)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=False, record_metrics=True, )
    return envs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--render_mode", type=str, default="all")
    args = parser.parse_args()

    env = make_mani_env(args.env_id, args.num_envs, render_mode=args.render_mode)
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

