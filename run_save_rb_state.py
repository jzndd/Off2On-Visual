import numpy as np
import cv2
import os
import shutil
import imageio
import metaworld
import os
import mujoco

import numpy as np
from rl_agent.utils import OfflineReplaybuffer
os.environ['MUJOCO_GL'] = 'egl'


def make_env(seed=4, max_path_length=500, **kwargs):
    st0 = np.random.get_state()
    np.random.seed(seed)
    env = Env(**kwargs)
    env.model.vis.global_.offwidth = 84
    env.model.vis.global_.offheight = 84
    # Ensure every time update, get different intial state
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    # the same seed can get the same result
    env._freeze_rand_vec = True
    np.random.set_state(st0)

    # SET CAMERA NAME
    env.mujoco_renderer.camera_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "corner2",
        )
    env.mujoco_renderer.width = 84
    env.mujoco_renderer.height = 84
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]

    env.max_path_length = max_path_length

    return env

def gen_traj(policy, seed, env_name, ep_num, dataset_name, max_path_length=100, save_data=False, use_random=True):
    action_repeat = 2
    env = make_env(seed=seed, max_path_length=max_path_length, render_mode="rgb_array")
    obs, info = env.reset()
    ret = 0

    seq_num = 1
    steps = 0

    rb = OfflineReplaybuffer(capacity=5000, obs_shape=(39,))
    # eps_data = []
    eps_states = []
    eps_actions = []
    eps_rewards = []
    eps_dones = []
    eps_next_states = []

    while True:
        if seq_num > ep_num:
            # print(f'max:{max}')
            # print(f'min:{min}')
            break

        if use_random:
            action = env.action_space.sample()
        else:
            action = policy.get_action(obs)
        eps_states.append(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        reward = 0.0
        for _ in range(action_repeat):
            obs, r, terminated, truncated, info = env.step(action)
            # reward += float(info['success'])
            reward += r
        ret += reward
        steps += 1
        done = terminated or truncated or bool(info['success'])

        eps_actions.append(action)
        eps_rewards.append(reward)
        eps_dones.append(done)
        eps_next_states.append(obs)

        # if success_num > success_num_the:
        if done:
            if info['success'] or use_random:
                assert len(eps_states) == len(eps_actions)
                for states, action, next_states, reward, done in zip(eps_states, eps_actions, eps_next_states, eps_rewards, eps_dones
                    ):
                    rb.store(states, action, next_states, reward, done)

                eps_states = []
                eps_actions = []
                eps_rewards = []
                eps_next_states = []
                eps_dones = []
                seq_num += 1
                seed += 1

                print(f"{env_name}, {seq_num}, Episode return: {ret}, traj_num: {steps}")
                print(f"info['success]: {info['success']}")

                env = make_env(seed=seed, max_path_length=max_path_length, render_mode="rgb_array")
                # print(f'{env_name}{env.action_space.low}')
                # print(f'{env_name}{env.action_space.high}')
                obs, info = env.reset()
                ret = 0
                steps = 0

            else:
                eps_states = []
                eps_actions = []
                eps_rewards = []
                eps_next_states = []
                eps_dones = []
                seed += 1
                steps = 0
                env = make_env(seed=seed, max_path_length=max_path_length, render_mode="rgb_array")
                obs, _ = env.reset()

                print(f"Failed {env_name}, {seq_num}, Episode return: {ret}")

    os.makedirs(f"{dataset_name}/{env_name}", exist_ok=True)
    rb.save(f"{dataset_name}/{env_name}/expert_rb_with_reward_state.pkl")
    print(rb.size)

        # if terminated or truncated:
        #     # if info["success"]==False:
        #     #     delete_path = f"{str(dataset_name)}/{str(env_name)}/{int(seq_num)}"
        #     #     shutil.rmtree(delete_path)
        #     success_num = 0
        #     img_num = 0
        #     obs, info = env.reset()
        #     print("-------------------------------")
        #     print(f"{env_name}, {seq_num}, Failed episode return: {ret}")
        #     print("-------------------------------")
        #     ret = 0

    # actions_path = f"{str(dataset_name)}/{str(env_name)}/actions.npy"
    # os.makedirs(os.path.dirname(actions_path), exist_ok=True)



if __name__ == "__main__":

    traj_num = 50
    dataset_name = f"data/"
    seed = list(range(traj_num)) 
    env_seed = 1500
    save_data = True
    use_random = False

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerNutAssemblyEnvV2 as Env       # Failed
    # from metaworld.policies import SawyerAssemblyV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'assembly-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBasketballEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerBasketballV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'basketball-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBinPickingEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerBinPickingV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'binpicking-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBoxCloseEnvV2 as Env           
    # from metaworld.policies import  SawyerBoxCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'boxclose-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressTopdownEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressTopdownV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-topdown-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressTopdownWallEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressTopdownWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-top-down-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressWallEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)


    # # coffee-button-v2                                                                     Successful
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeeButtonEnvV2 as Env
    # from metaworld.policies import SawyerCoffeeButtonV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-button-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeePullEnvV2 as Env
    # from metaworld.policies import SawyerCoffeePullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-pull-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # coffee-push-v2
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeePushEnvV2 as Env
    # from metaworld.policies import SawyerCoffeePushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-push-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDialTurnEnvV2 as Env          # Failed
    # from metaworld.policies import SawyerDialTurnV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'dial-turn-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerNutDisassembleEnvV2 as Env          # Failed
    from metaworld.policies import SawyerDisassembleV2Policy as EnvPolicy
    policy = EnvPolicy()
    env_name = 'disassemble-v2'
    gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorCloseEnvV2 as Env          # Failed
    # from metaworld.policies import SawyerDoorCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'door-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorLockEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerDoorLockV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'door-lock-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorUnlockEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerDoorUnlockV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'door-unlock-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDrawerCloseEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerDrawerCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'drawer-close-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDrawerOpenEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerDrawerOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'drawer-open-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerFaucetCloseEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerFaucetCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'faucet-close-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerFaucetOpenEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerFaucetOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'faucet-open-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)
    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHammerEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHammerV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'hammer-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandInsertEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandInsertV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'hand-insert-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePressSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePressSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-press-side-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePressEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePressV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-press-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePullSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePullSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-pull-side-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePullEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-pull-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerLeverPullEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerLeverPullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'lever-pull-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPegInsertionSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPegInsertionSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'peg-insertion-side-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPegUnplugSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPegUnplugSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'peg-unplug-side-v2'
    # gen_traj(policy, env, env_name,traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickOutOfHoleEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickOutOfHoleV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-out-of-hole-v2'
    # gen_traj(policy, env,env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickPlaceEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickPlaceV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-place-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)
    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickPlaceWallEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickPlaceWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-place-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideBackSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPlateSlideBackSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-back-side-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideBackEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPlateSlideBackV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-back-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideSideEnvV2 as Env
    # from metaworld.policies import SawyerPlateSlideSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-side-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPlateSlideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushBackEnvV2 as Env
    # from metaworld.policies import SawyerPushBackV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-back-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushEnvV2 as Env
    # from metaworld.policies import SawyerPushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushWallEnvV2 as Env
    # from metaworld.policies import SawyerPushWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachEnvV2 as Env
    # from metaworld.policies import SawyerReachV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'reach-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachWallEnvV2 as Env
    # from metaworld.policies import SawyerReachWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'reach-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerShelfPlaceEnvV2 as Env
    # from metaworld.policies import SawyerShelfPlaceV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'shelf-place-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSoccerEnvV2 as Env
    # from metaworld.policies import SawyerSoccerV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'soccer-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerStickPullEnvV2 as Env
    # from metaworld.policies import SawyerStickPullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'stick-pull-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerStickPushEnvV2 as Env
    # from metaworld.policies import SawyerStickPushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'stick-push-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSweepEnvV2 as Env              # Failed
    # from metaworld.policies import SawyerSweepV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'sweep-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSweepIntoGoalEnvV2 as Env              # Failed
    # from metaworld.policies import SawyerSweepIntoV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'sweep-into-goal-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    # window-close-v2
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerWindowCloseEnvV2 as Env
    # from metaworld.policies import SawyerWindowCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'window-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerWindowOpenEnvV2 as Env
    # from metaworld.policies import SawyerWindowOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'window-open-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name)