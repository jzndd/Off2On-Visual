from collections import deque
import numpy as np
import cv2
import os
import shutil
import imageio
import metaworld
import os
import mujoco
import torch

import numpy as np
from rl_agent.utils import OfflineReplaybuffer
os.environ['MUJOCO_GL'] = 'egl'


def make_env(seed=4, max_path_length=500, img_size=128, **kwargs):
    st0 = np.random.get_state()
    np.random.seed(seed)
    env = Env(**kwargs)
    env.model.vis.global_.offwidth = img_size
    env.model.vis.global_.offheight = img_size
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
    env.mujoco_renderer.width = img_size
    env.mujoco_renderer.height = img_size
    env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]

    env.max_path_length = max_path_length

    return env

def get_stack_obs(obs_buffer):
    """
    Stack the last frame_stack frames together
    """
    stack_obs = np.concatenate(list(obs_buffer), axis=-1) #  84 * 84 * (3 * framestack)
    stack_obs = torch.as_tensor(stack_obs, device="cuda").div(255).mul(2).sub(1).permute(2,0,1).contiguous().detach().cpu().numpy()
    return stack_obs

def gen_traj(policy, seed, env_name, ep_num, dataset_name, max_path_length=100, 
             save_data=False, use_random=True, 
             use_sparse_reward=True, save_whole_traj=False,
             img_size=96, action_repeat=2,
             frame_stack=3):
    env = make_env(seed=seed, max_path_length=max_path_length, img_size=img_size, render_mode="rgb_array")
    obs, info = env.reset()
    ret = 0

    seq_num = 1
    steps = 0
    fail_num = 0

    rb = OfflineReplaybuffer(capacity=5000, obs_shape=(3 * frame_stack, img_size, img_size))
    # eps_data = []
    eps_imgs = []
    eps_actions = []
    eps_rewards = []
    eps_dones = []
    eps_next_imgs = []

    obs_buffer = deque([], maxlen=frame_stack)

    success_flag = False

    while True:
        if seq_num > ep_num:
            # print(f'max:{max}')
            # print(f'min:{min}')
            break

        if fail_num > 50 and seq_num < 2:
            break
        

        ## access the current img
        env.camera_name="corner2"
        cam_img = env.render()
        cam_img = np.flipud(cam_img).copy()

        if steps == 0:
            for _ in range(frame_stack):
                obs_buffer.append(cam_img)
        
        stack_obs = get_stack_obs(obs_buffer)
        eps_imgs.append(stack_obs)

        ## sample action

        if use_random:
            action = env.action_space.sample()
        else:
            action = policy.get_action(obs)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        reward = 0.0

        for _ in range(action_repeat):
            obs, r, terminated, truncated, info = env.step(action)
            reward += r
        
        if use_sparse_reward:
            reward = 1.0 if info["success"] else 0.0

        ret += reward
        steps += 1

        if save_whole_traj:
            done = terminated or truncated
        else:
        # done = terminated or truncated
            done = terminated or truncated or info['success']

        success_flag |= bool(info['success'])

        eps_actions.append(action)
        eps_rewards.append(reward)
        eps_dones.append(done)

        env.camera_name="corner2"
        cam_img = env.render()
        cam_img = np.flipud(cam_img).copy()
        
        obs_buffer.append(cam_img)
        stack_next_obs = get_stack_obs(obs_buffer)
        eps_next_imgs.append(stack_next_obs)

        # if success_num > success_num_the:
    
        if done:
            if success_flag or use_random:
                # remove the last image
                # eps_imgs = eps_imgs[1:]
                # print("len of eps_imgs", len(eps_imgs))
                # print("len of eps_actions", len(eps_actions))
                # exit(1)
                assert len(eps_imgs) == len(eps_actions)
                for img, action, next_img, reward, done in zip(eps_imgs, eps_actions, eps_next_imgs, eps_rewards, eps_dones
                    ):
                    rb.store(img, action, next_img, reward, done)

                # import pdb; pdb.set_trace()

                eps_imgs = []
                eps_actions = []
                eps_rewards = []
                eps_next_imgs = []
                eps_dones = []
                seq_num += 1
                seed += 1
                env = make_env(seed=seed, max_path_length=max_path_length, img_size=img_size, render_mode="rgb_array")
                obs, info = env.reset()

                print(f"{env_name}, {seq_num}, Episode return: {ret}, traj_num: {steps}")
                ret = 0
                steps = 0

                success_flag = False
            else:
                eps_imgs = []
                eps_actions = []
                eps_rewards = []
                eps_next_imgs = []
                eps_dones = []
                seed += 1
                fail_num += 1
                env = make_env(seed=seed, max_path_length=max_path_length, img_size=img_size, render_mode="rgb_array")
                obs, _ = env.reset()

                print(f"Failed {env_name}, {seq_num}, Episode return: {ret}, step: {steps}")
                ret = 0
                steps = 0
                success_flag = False

    if fail_num > 50 and seq_num < 2:
        return None

    file_name = "expert_rb_with_reward"
    if use_sparse_reward:
        file_name += "_sparse"
    
    if save_whole_traj:
        file_name += "_whole_traj_50traj"

    if frame_stack > 1:
        file_name += f"_stack{frame_stack}"

    file_name += ".pkl"

    os.makedirs(f"{dataset_name}/{env_name}_{img_size}", exist_ok=True)
    rb.save(f"{dataset_name}/{env_name}_{img_size}/{file_name}")

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
    use_sparse_reward = True
    save_whole_traj = True
    img_size = 96
    frame_stack = 3

# handle-pull-side-v2, door-lock-v2, lever-pull-v2, coffee-pull-v2 

# handle-press-v2（92%），lever-pull-v2（56%），window-open-v2 （76%），plate-slide-v2 （60%），soccer-v2 （8%），handle-pull-side-v2（56%），handle-pull-v2（48% ），disassemble-v2(36%), coffee-pull-v2（20%），hammer-v2（56%）

# door-close-v2, door-lock-v2, door-unlock-v2
# basketball-v2 ,peg-unplug-side-v2, plate-slide-back-v2, push-back-v2, shelf-place-v2

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerNutAssemblyEnvV2 as Env       # Failed
    # from metaworld.policies import SawyerAssemblyV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'assembly-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBasketballEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerBasketballV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'basketball-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBinPickingEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerBinPickingV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'binpicking-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerBoxCloseEnvV2 as Env           
    # from metaworld.policies import  SawyerBoxCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'boxclose-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # Always win
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressTopdownEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressTopdownV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-topdown-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # Always win
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressTopdownWallEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressTopdownWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-topdown-wall-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # Always win
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, use_random=use_random)

    # Always win
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerButtonPressWallEnvV2 as Env           
    # from metaworld.policies import  SawyerButtonPressWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'button-press-wall-v2'
    # gen_traj(policy, env, env_name, traj_num, dataset_name)

    #                                                                    Successful
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeeButtonEnvV2 as Env
    # from metaworld.policies import SawyerCoffeeButtonV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-button-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeePullEnvV2 as Env
    # from metaworld.policies import SawyerCoffeePullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-pull-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerCoffeePushEnvV2 as Env        # Failed
    # from metaworld.policies import SawyerCoffeePushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'coffee-push-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDialTurnEnvV2 as Env          # Failed
    # from metaworld.policies import SawyerDialTurnV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'dial-turn-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size= img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerNutDisassembleEnvV2 as Env          # Success
    # from metaworld.policies import SawyerDisassembleV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'disassemble-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorCloseEnvV2 as Env          # Failed
    # from metaworld.policies import SawyerDoorCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'door-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorLockEnvV2 as Env           # Failed
    from metaworld.policies import  SawyerDoorLockV2Policy as EnvPolicy
    policy = EnvPolicy()
    env_name = 'door-lock-v2'
    gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
             use_random=use_random, 
             use_sparse_reward=use_sparse_reward,
             save_whole_traj=save_whole_traj,
             frame_stack=frame_stack)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDoorUnlockEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerDoorUnlockV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'door-unlock-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDrawerCloseEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerDrawerCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'drawer-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerDrawerOpenEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerDrawerOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'drawer-open-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerFaucetCloseEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerFaucetCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'faucet-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerFaucetOpenEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerFaucetOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'faucet-open-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj)
    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHammerEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHammerV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'hammer-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)
    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandInsertEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandInsertV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'hand-insert-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePressSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePressSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-press-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePressEnvV2 as Env           # Cuccess
    # from metaworld.policies import  SawyerHandlePressV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-press-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePullSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePullSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-pull-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerHandlePullEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerHandlePullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'handle-pull-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerLeverPullEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerLeverPullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'lever-pull-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPegInsertionSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPegInsertionSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'peg-insertion-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPegUnplugSideEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPegUnplugSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'peg-unplug-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)


    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickOutOfHoleEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickOutOfHoleV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-out-of-hole-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickPlaceEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickPlaceV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-place-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)
    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPickPlaceWallEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPickPlaceWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'pick-place-wall-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideBackSideEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerPlateSlideBackSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-back-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideBackEnvV2 as Env           # Failed
    # from metaworld.policies import  SawyerPlateSlideBackV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-back-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideSideEnvV2 as Env
    # from metaworld.policies import SawyerPlateSlideSideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-side-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPlateSlideEnvV2 as Env           # Success
    # from metaworld.policies import  SawyerPlateSlideV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'plate-slide-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    
    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushBackEnvV2 as Env
    # from metaworld.policies import SawyerPushBackV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-back-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushEnvV2 as Env
    # from metaworld.policies import SawyerPushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerPushWallEnvV2 as Env
    # from metaworld.policies import SawyerPushWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'push-wall-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachEnvV2 as Env
    # from metaworld.policies import SawyerReachV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'reach-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerReachWallEnvV2 as Env
    # from metaworld.policies import SawyerReachWallV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'reach-wall-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerShelfPlaceEnvV2 as Env
    # from metaworld.policies import SawyerShelfPlaceV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'shelf-place-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSoccerEnvV2 as Env
    # from metaworld.policies import SawyerSoccerV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'soccer-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerStickPullEnvV2 as Env
    # from metaworld.policies import SawyerStickPullV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'stick-pull-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerStickPushEnvV2 as Env  # Success
    # from metaworld.policies import SawyerStickPushV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'stick-push-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSweepEnvV2 as Env              # Failed
    # from metaworld.policies import SawyerSweepV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'sweep-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerSweepIntoGoalEnvV2 as Env              # Failed
    # from metaworld.policies import SawyerSweepIntoV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'sweep-into-goal-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerWindowCloseEnvV2 as Env   # Success
    # from metaworld.policies import SawyerWindowCloseV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'window-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerWindowOpenEnvV2 as Env # Success
    # from metaworld.policies import SawyerWindowOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'window-open-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)

    # from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerWindowOpenEnvV2 as Env # Success
    # from metaworld.policies import SawyerWindowOpenV2Policy as EnvPolicy
    # policy = EnvPolicy()
    # env_name = 'box-close-v2'
    # gen_traj(policy, env_seed, env_name, traj_num, dataset_name, save_data=save_data, 
    #          use_random=use_random, 
    #          use_sparse_reward=use_sparse_reward,
    #          save_whole_traj=save_whole_traj,
    #          img_size=img_size)