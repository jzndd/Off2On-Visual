from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union

from omegaconf import ListConfig
import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

# from coroutines.env_loop import make_env_loop
# from envs import TorchEnv, WorldModelEnv
# from utils import init_lstm, LossAndLogs

import os
import sys

from .utils import *
from torch.distributions import Normal
from .baseagent import *
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler  

from .net import *

from copy import deepcopy

@dataclass
class PPOConfig:
    clip_param: float = 0.2
    K_epochs: int = 10
    mini_batch_size: int = 256
    max_grad_norm: float = 0.5
    adv_compute_mode: str = "tradition" # optional:"tradition", "gae", "q-v" 
    use_adam_eps: bool = True
    use_std_share_network: bool = False
    use_lr_decay: bool = True
    use_state_norm: bool = True

class PPOAgent2D(BaseAgent):
    def __init__(
        self,
        cfg: ActorCriticConfig,
        ppo_cfg: PPOConfig,
        use_old_policy: bool = True,
        # adv_compute_mode: str = "tradition", # optional:"tradition", "gae", "q-v", "iql"
    ):
        super().__init__(cfg)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## initialize ppo hyperparameters
        self.clip_param =  ppo_cfg.clip_param
        self.K_epochs =  ppo_cfg.K_epochs
        self.mini_batch_size =  ppo_cfg.mini_batch_size
        self.max_grad_norm =  ppo_cfg.max_grad_norm
        # state_dim = cfg.num_states
        self.gae_lambda = 0.95

        self.ppo_cfg = ppo_cfg
        ## network initialization

        self.online_lr = cfg.online_lr

        self.encoder = ActorCriticEncoder(cfg)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=1e-4)

        self.actor = Actorlog(cfg, self.encoder.repr_dim, use_trunk=True, use_std_share_network=ppo_cfg.use_std_share_network)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.adv_compute_mode = ppo_cfg.adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, self.encoder.repr_dim, use_trunk=True)
            self.value_optim = optim.Adam(self.value.parameters(), lr=1e-4)
            if self.adv_compute_mode == 'q-v':
                self.doubleq = DoubleQMLP(cfg, self.encoder.repr_dim, use_trunk=True)
                self.doubleq_optim = optim.Adam(self.doubleq.parameters(), lr=1e-4)
        elif self.adv_compute_mode == 'iql':
            self.critic = IQLCritic(cfg, self.encoder.repr_dim, use_trunk=True) 
        else:
            raise NotImplementedError

        self.ac_type = "ppo"
    
    @torch.no_grad()
    def predict_act(self, obs: torch.Tensor, eval_mode=False) -> ActorCriticOutput:
        assert obs.ndim == 4  # Ensure observation shape is correct
        h = self.encoder(obs).flatten(start_dim=1)
        if self.adv_compute_mode == 'tradition' and not eval_mode:
            # When tradition, return both action and state_value
            return *self.actor.get_action(h, eval_mode=eval_mode), self.value(obs)
        return self.actor.get_action(h, eval_mode=eval_mode)

    def bc_actor_update(self, rb: OfflineReplaybuffer):

        c = self.loss_cfg
        bacth = rb.sample(mini_batch_size=256)

        obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)

        h = self.encoder(obs).flatten(start_dim=1)

        dist = self.actor.get_dist(h)
        log_prob = dist.log_prob(act)
        bc_loss = -log_prob.sum(dim=-1).mean() 
        bc_loss = c.weight_bc_loss * bc_loss

        self.actor_optim.zero_grad()
        self.encoder_optim.zero_grad()
        bc_loss.backward()
        self.encoder_optim.step()
        self.actor_optim.step()

    def bc_critic_update(self, rb: OfflineReplaybuffer):

        bacth = rb.sample(mini_batch_size=256)

        obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)

        with torch.no_grad():
            h = self.encoder(obs).flatten(start_dim=1)
            h_ = self.encoder(obs_).flatten(start_dim=1)

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            val = self.value(h)
            value_loss = F.mse_loss(val, returns)

            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

            if self.adv_compute_mode == 'q-v':
                with torch.no_grad():
                    act_ = self.actor.get_action(h_, eval_mode=True)
                    target_q = reward + (1-done) * 0.99 * torch.min(*self.doubleq(h_, act_))
                q1, q2 = self.doubleq(h, act)
                q_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)

                self.doubleq_optim.zero_grad()
                q_loss.backward()
                self.doubleq_optim.step()

        elif self.adv_compute_mode == 'iql':
            self.critic.update(h, act, reward, 1-done, h_)
    
    def bc_transfer_ac(self):

        del self.actor_optim

        params_list = list(self.actor.parameters())
        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            del self.value_optim
            params_list += list(self.value.parameters())
            if self.adv_compute_mode == 'q-v':
                del self.doubleq_optim
                params_list += list(self.doubleq.parameters())
        elif self.adv_compute_mode == 'iql':
            self.critic.transbc2online(self.online_lr)

        if self.ppo_cfg.use_adam_eps:
            self.optimizer = optim.Adam(params_list, lr=self.online_lr, eps=1e-5)
        else:
            self.optimizer = optim.Adam(params_list, lr=self.online_lr)

        self.fix_encoder = True
        self.encoder.requires_grad_(False)
    
    def update(self, rb: OnlineReplayBuffer, iter=None, max_iter=None):

        c = self.loss_cfg
    
        batch = rb.sample_all()
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_values, dws = to_torch(batch, self.device)

        with torch.no_grad():
            obss = self.encoder(obss).flatten(start_dim=1)
            next_obss = self.encoder(next_obss).flatten(start_dim=1)

        # old_log_probs = old_log_probs.squeeze(-1)
        old_state_values = old_state_values.squeeze(-1) if old_state_values is not None else None

        returns = compute_returns(rews, 0.99, dones)
        returns = returns.to(self.device)
        returns = returns.squeeze(-1)

        with torch.no_grad():
            if self.adv_compute_mode == 'tradition':  
                # rerference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L221
                advantages = returns.detach() - old_state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            elif self.adv_compute_mode == 'q-v':
                advantages = torch.min(*self.doubleq(obss, acts)) - self.value(obss)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            elif self.adv_compute_mode == 'gae':
                adv = []
                gae = 0
                vs = self.value(obss)
                next_vs = self.value(next_obss)
                deltas = rews + 0.99 * (1.0 - dws) * next_vs - vs
                for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                    gae = delta + 0.99 * self.gae_lambda * gae * (1.0 - d)
                    adv.insert(0, gae)
                advantages = torch.tensor(adv, dtype=torch.float32).view(-1,1).to(self.device)
                v_target = advantages + vs
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            elif self.adv_compute_mode == 'iql':
                advantages = self.critic.get_advantage(obss, acts)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        batch_size = rb.size

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                # if not self.use_old_policy:
                dist = self.actor.get_dist(obss[index])
                log_prob = dist.log_prob(acts[index]).sum(dim=1, keepdim=True)

                if self.adv_compute_mode in ["tradition", "q-v"]:
                    state_value = self.value(obss[index].detach())
                    value_loss = F.mse_loss(state_value.squeeze(), returns[index])
                    q_loss = torch.tensor(0.0)
                    params_list = list(self.value.parameters())
                    if self.adv_compute_mode == 'q-v':
                        with torch.no_grad():
                            next_act = self.actor.get_action(next_obss[index], eval_mode=True)
                            target_q = rews[index] + (1-dones[index]) * 0.99 * torch.min(*self.doubleq(next_obss[index], next_act))
                        q1, q2 = self.doubleq(obss[index], acts[index])
                        q_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)
                        params_list += list(self.doubleq.parameters())
                elif self.adv_compute_mode == 'gae':
                    params_list = list(self.value.parameters())
                    state_value = self.value(obss[index].detach())
                    value_loss = F.mse_loss(v_target[index], state_value)
                    q_loss = torch.tensor(0.0)
                elif self.adv_compute_mode == "iql":
                    params_list = []
                    value_loss = torch.tensor(0.0)
                    q_loss = torch.tensor(0.0)
                    self.critic.update(obss[index], acts[index], rews[index], 1-dones[index], next_obss[index])

                ratios = torch.exp(log_prob - old_log_probs[index].sum(1, keepdim=True).detach())
        
                entropy_loss = -c.weight_entropy_loss * dist.entropy().mean()
                surrogate = -advantages[index] * ratios
                surrogate_clipped = -advantages[index] * torch.clamp(ratios,
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                
                policy_loss = torch.max(surrogate, surrogate_clipped).mean() + entropy_loss

                loss = policy_loss + 0.5 * value_loss + 0.5 * q_loss
        
                self.optimizer.zero_grad()
                loss.backward()
                params_list += list(self.actor.parameters())
                nn.utils.clip_grad_norm_(params_list, self.max_grad_norm)
                self.optimizer.step()
        
        if self.ppo_cfg.use_lr_decay and iter is not None and max_iter is not None:
            self.lr_decay(iter, max_iter)
            
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.item(),
                "surrogate": surrogate.mean().item(), "surrogate_clipped": surrogate_clipped.mean().item()} 
        
    def lr_decay(self, step, max_step):
        lr = self.online_lr * (1 - step / max_step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr

class PPOAgent(BaseAgent):
    def __init__(
        self,
        cfg: ActorCriticConfig,
        ppo_cfg: PPOConfig,
        use_old_policy: bool = True,
        # adv_compute_mode: str = "tradition", # optional:"tradition", "gae", "q-v", "iql"
    ):
        super().__init__(cfg)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## initialize ppo hyperparameters
        self.clip_param =  ppo_cfg.clip_param
        self.K_epochs =  ppo_cfg.K_epochs
        self.mini_batch_size =  ppo_cfg.mini_batch_size
        self.max_grad_norm =  ppo_cfg.max_grad_norm
        state_dim = cfg.num_states
        self.gae_lambda = 0.95

        self.ppo_cfg = ppo_cfg
        ## network initialization

        self.online_lr = cfg.online_lr

        self.actor = Actorlog(cfg, state_dim, use_trunk=False, use_std_share_network=ppo_cfg.use_std_share_network)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.adv_compute_mode = ppo_cfg.adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, state_dim, use_trunk=False)
            self.value_optim = optim.Adam(self.value.parameters(), lr=3e-4)
            if self.adv_compute_mode == 'q-v':
                self.doubleq = DoubleQMLP(cfg, state_dim, use_trunk=False)
                self.doubleq_optim = optim.Adam(self.doubleq.parameters(), lr=3e-4)
        elif self.adv_compute_mode == 'iql':
            self.critic = IQLCritic(cfg, state_dim, use_trunk=False) 
        else:
            raise NotImplementedError

        self.ac_type = "ppo"
        
    def predict_act(self, obs: torch.Tensor, eval_mode=False) -> ActorCriticOutput:
        assert obs.ndim == 2  # Ensure observation shape is correct
        if self.adv_compute_mode == 'tradition' and not eval_mode:
            # When tradition, return both action and state_value
            return *self.actor.get_action(obs, eval_mode=eval_mode), self.value(obs)
        return self.actor.get_action(obs, eval_mode=eval_mode)

    def bc_actor_update(self, rb: OfflineReplaybuffer):

        c = self.loss_cfg
        bacth = rb.sample(mini_batch_size=256)

        obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)

        dist = self.actor.get_dist(obs)
        log_prob = dist.log_prob(act)
        bc_loss = -log_prob.sum(dim=-1).mean() 
        bc_loss = c.weight_bc_loss * bc_loss

        self.actor_optim.zero_grad()
        bc_loss.backward()
        self.actor_optim.step()

    def bc_critic_update(self, rb: OfflineReplaybuffer):

        c = self.loss_cfg
        bacth = rb.sample(mini_batch_size=256)

        obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            val = self.value(obs)
            value_loss = F.mse_loss(val, returns)

            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

            if self.adv_compute_mode == 'q-v':
                with torch.no_grad():
                    act_ = self.actor.get_action(obs_, eval_mode=True)
                    target_q = reward + (1-done) * 0.99 * torch.min(*self.doubleq(obs_, act_))
                q1, q2 = self.doubleq(obs, act)
                q_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)

                self.doubleq_optim.zero_grad()
                q_loss.backward()
                self.doubleq_optim.step()

        elif self.adv_compute_mode == 'iql':
            self.critic.update(obs, act, reward, 1-done, obs_)
    
    def bc_transfer_ac(self):

        if self.ppo_cfg.use_adam_eps:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.online_lr, eps=1e-5)
        else:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.online_lr)

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            del self.value_optim
            params_list = list(self.value.parameters())
            if self.adv_compute_mode == 'q-v':
                del self.doubleq_optim
                params_list += list(self.doubleq.parameters())
            self.critic_optim = optim.Adam(params_list, lr=self.online_lr, eps=1e-5)
        elif self.adv_compute_mode == 'iql':
            self.critic.transbc2online(self.online_lr)
    
    def update(self, rb: OnlineReplayBuffer, iter=None, max_iter=None):

        c = self.loss_cfg
    
        batch = rb.sample_all()
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_values, dws = to_torch(batch, self.device)

        # old_log_probs = old_log_probs.squeeze(-1)
        old_state_values = old_state_values.squeeze(-1) if old_state_values is not None else None

        returns = compute_returns(rews, 0.99, dones)
        returns = returns.to(self.device)
        returns = returns.squeeze(-1)

        with torch.no_grad():
            if self.adv_compute_mode == 'tradition':  
                # rerference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/728cce83d7ab628fe2634eabcdf3239997eb81dd/PPO.py#L221
                advantages = returns.detach() - old_state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            elif self.adv_compute_mode == 'q-v':
                advantages = torch.min(*self.doubleq(obss, acts)) - self.value(obss)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            elif self.adv_compute_mode == 'gae':
                adv = []
                gae = 0
                vs = self.value(obss)
                next_vs = self.value(next_obss)
                deltas = rews + 0.99 * (1.0 - dws) * next_vs - vs
                for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                    gae = delta + 0.99 * self.gae_lambda * gae * (1.0 - d)
                    adv.insert(0, gae)
                advantages = torch.tensor(adv, dtype=torch.float32).view(-1,1).to(self.device)
                v_target = advantages + vs
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            elif self.adv_compute_mode == 'iql':
                advantages = self.critic.get_advantage(obss, acts)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        batch_size = rb.size

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                # if not self.use_old_policy:
                dist = self.actor.get_dist(obss[index])
                log_prob = dist.log_prob(acts[index])

                if self.adv_compute_mode in ["tradition", "q-v"]:
                    state_value = self.value(obss[index].detach())
                    value_loss = F.mse_loss(state_value.squeeze(), returns[index])
                    q_loss = torch.tensor(0.0)
                    params_list = list(self.value.parameters())
                    if self.adv_compute_mode == 'q-v':
                        with torch.no_grad():
                            next_act = self.actor.get_action(next_obss[index], eval_mode=True)
                            target_q = rews[index] + (1-dones[index]) * 0.99 * torch.min(*self.doubleq(next_obss[index], next_act))
                        q1, q2 = self.doubleq(obss[index], acts[index])
                        q_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)
                        params_list += list(self.doubleq.parameters())
                elif self.adv_compute_mode == 'gae':
                    params_list = list(self.value.parameters())
                    state_value = self.value(obss[index])
                    value_loss = F.mse_loss(state_value, v_target[index])
                    q_loss = torch.tensor(0.0)
                elif self.adv_compute_mode == "iql":
                    value_loss = torch.tensor(0.0)
                    q_loss = torch.tensor(0.0)
                    self.critic.update(obss[index], acts[index], rews[index], 1-dones[index], next_obss[index])

                ratios = torch.exp(log_prob.sum(1, keepdim=True) - old_log_probs[index].sum(1, keepdim=True).detach())
        
                entropy_loss = -c.weight_entropy_loss * dist.entropy().sum(1, keepdim=True)
                surrogate = -advantages[index] * ratios
                surrogate_clipped = -advantages[index] * torch.clamp(ratios,
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                
                policy_loss = torch.max(surrogate, surrogate_clipped) + entropy_loss
                policy_loss = policy_loss.mean()

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                if self.adv_compute_mode != 'iql':
                    critic_loss = value_loss + q_loss
            
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(params_list, self.max_grad_norm)
                    self.critic_optim.step()
        
        if self.ppo_cfg.use_lr_decay and iter is not None and max_iter is not None:
            self.lr_decay(iter, max_iter)
            
        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.mean().item(),
                "surrogate": surrogate.mean().item(), "surrogate_clipped": surrogate_clipped.mean().item()} 
        
    def lr_decay(self, step, max_step):
        lr = self.online_lr * (1 - step / max_step)
        for p in self.actor_optim.param_groups:
            p['lr'] = lr
        for p in self.critic_optim.param_groups:
            p['lr'] = lr