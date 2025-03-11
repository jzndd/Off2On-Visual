### run script

```python
python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256

python main.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128
```

### 2D
```python
python main2D.py common.devices=[1] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128 task=button-press-topdown-v2 agent.ppo_cfg.adv_compute_mode=gae common.seed=42

python main2D.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=256 task=button-press-topdown-v2

python main2D.py common.devices=[3] ac_type=ppo actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128 task=button-press-topdown-v2
```


### dmc control 

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=iql agent.ppo_cfg.clip_param=0.2 common.devices=[7]

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=128 agent.ppo_cfg.adv_compute_mode=iql common.devices=[6] agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=128 common.devices=[5] agent.ppo_cfg.adv_compute_mode=gae agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=512 agent.ppo_cfg.mini_batch_size=128 common.devices=[4] agent.ppo_cfg.adv_compute_mode=gae agent.ppo_cfg.clip_param=0.2

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=1024 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=iql common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[6] agent.actor_critic_cfg.hidden_dim=64 common.seed=42



python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[0] agent.actor_critic_cfg.hidden_dim=64 common.seed=42 agent.actor_critic_cfg.online_lr=3e-4

python main.py task=walker2d-medium-v2 env=dmc actor_critic.training.batch_size=2048 agent.ppo_cfg.mini_batch_size=64 agent.ppo_cfg.adv_compute_mode=gae common.devices=[1] agent.actor_critic_cfg.hidden_dim=256 common.seed=42



