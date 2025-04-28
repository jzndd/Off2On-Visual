### 按 policy 分布收集数据的一般流程


+ 收集纯 expert 数据以训 bc
```bash
# 请调整 img_size 参数 ！以及可能的 action_repeat 参数
# 其他参数说明：save_whole_traj （是否保存完整轨迹，训 world model 需要保存完整轨迹，训policy 不需要保存）
# use_sparse_reward 恒定为 True
# use_random 恒定为 False
python run_save_rb.py
```

+ train 一个 bc policy
```bash
python main2D.py only_bc=True img_size=128
```
训练好后将 outputs/...../bc.pth 手动放到 ckpt/..../bc.pth 中

+ 利用 bc policy 收集数据
```bash
python main2D.py save_data=True img_size=96
```  

启动save_data 时，会执行  

```python
trainer = Trainer(cfg, root_dir)
trainer.save_data()                 # save raw data (用于训练) 
trainer.save_eval_data()            # save test data (包括了 raw data 和 images 和 video， 去对应的 outputs/ 文件夹中 move 出来就行)
```

### ppo
#### for metaworld
```bash
python main2D.py common.devices=[1] seed=100 ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=door-lock-v2 agent.ppo_cfg.adv_compute_mode=iql2gae
```

### vrl3 
#### for metaworld
```bash
python main2D_offpolicy.py common.devices=[0] seed=100 ac_type=vrl3 task=door-lock-v2 actor_critic.training.batch_size=512 is_sparse_reward=True agent.actor_critic_cfg.online_lr=1e-4

python main2D_offpolicy.py common.devices=[1] seed=100 ac_type=vrl3 task=door-lock-v2 actor_critic.training.batch_size=512 is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4
```

#### for dmc
```bash
python main2D_offpolicy.py common.devices=[3] common.seed=10 ac_type=vrl3 task=walker_walk actor_critic.training.batch_size=256 is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5 only_bc=True
```

### DrQv2+RLPD for metaworld

#### for metaworld
```bash
# dense reward  + RLPD
python main2D_offpolicy_v2.py common.devices=[0] seed=100 ac_type=rlpd task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 frame_stack=1 agent.drqv2_cfg.offline_data_ratio=0.5

python main2D_offpolicy_v2.py common.devices=[0] seed=100 ac_type=rlpd task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 frame_stack=3 agent.drqv2_cfg.offline_data_ratio=0.5

# sparse reward + RLPD
python main2D_offpolicy_v2.py common.devices=[1] seed=100 ac_type=rlpd task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=True agent.actor_critic_cfg.online_lr=1e-4 frame_stack=1 agent.drqv2_cfg.offline_data_ratio=0.5

python main2D_offpolicy_v2.py common.devices=[1] seed=100 ac_type=rlpd task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=True agent.actor_critic_cfg.online_lr=1e-4 frame_stack=3 agent.drqv2_cfg.offline_data_ratio=0.5

# dense reward + no_offline_data
python main2D_offpolicy_v2.py common.devices=[2] common.seed=100 ac_type=drqv2 task=door-lock-v2 actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 agent.drqv2_cfg.offline_data_ratio=0
```

#### for dmc
```bash
python main2D_offpolicy_v2.py common.devices=[0] common.seed=10 ac_type=drqv2 task=walker_walk actor_critic.training.batch_size=512 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5

python main2D_offpolicy_v2.py common.devices=[2] common.seed=10 ac_type=drqv2 task=walker_walk actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5 frame_stack=3

# online & only use online data
python main2D_offpolicy_v2.py common.devices=[1] common.seed=10 ac_type=drqv2 task=walker_walk actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5 frame_stack=3 agent.drqv2_cfg.offline_data_ratio=0

# online & use both online and offline data
python main2D_offpolicy_v2.py common.devices=[0] common.seed=10 ac_type=rlpd task=walker_walk actor_critic.training.batch_size=256 train_with_bc=False is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5 frame_stack=3 agent.drqv2_cfg.offline_data_ratio=0.5

# offline & only use offline data
python main2D_offpolicy_v2.py common.devices=[0] common.seed=10 ac_type=drqv2offline task=walker_walk actor_critic.training.batch_size=256 train_with_bc=True only_bc=True is_sparse_reward=False agent.actor_critic_cfg.online_lr=1e-4 img_size=84 evaluation.eval_times=5 frame_stack=3 agent.drqv2_cfg.bc_weight=2.5
```

### run script

```bash
python main.py task=walker2d-medium-v2 env=dmc common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae agent.actor_critic_cfg.hidden_dim=64 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4 training.bc_actor_warmup_steps=20000 training.bc_critic_warmup_steps=20000  

python main.py task=walker2d-medium-v2 env=dmc common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae agent.actor_critic_cfg.hidden_dim=256 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4 

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128
```

### 2D
```bash
python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True

python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 agent.actor_critic_cfg.online_lr=3e-5 is_sparse_reward=True

python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True agent.actor_critic_cfg.online_lr=3e-5

python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True

# whole traj
python main2D.py common.devices=[0] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=100 agent.actor_critic_cfg.hidden_dim=1024 is_sparse_reward=True agent.actor_critic_cfg.online_lr=3e-5

python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 task=hammer-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=100 agent.actor_critic_cfg.hidden_dim=1024 agent.actor_critic_cfg.online_lr=3e-5 is_sparse_reward=True

# save data
python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True save_data=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False 

# tradition
python main2D.py common.devices=[0] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=tradition actor_critic.training.batch_size=256 agent.ppo_cfg.mini_batch_size=256 seed=100

python main2D.py common.devices=[0] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=tradition actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 seed=200

# iql
python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=iql actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 seed=100

python main2D.py common.devices=[1] ac_type=ppo is_sparse_reward=True task=hammer-v2 agent.actor_critic_cfg.hidden_dim=1024  only_bc=False agent.ppo_cfg.adv_compute_mode=iql actor_critic.training.batch_size=256 agent.ppo_cfg.mini_batch_size=256 seed=200

# tradition
```

### dmc control 
```bash
python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=iql agent.ppo_cfg.clip_param=0.2 common.devices=[7]
```












