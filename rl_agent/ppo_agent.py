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
    use_bc: bool = False
    target_kl: float = 0.2
    clip_vloss: bool = True

class PPOAgent2D(BaseAgent):
    def __init__(
        self,
        cfg: ActorCriticConfig,
        ppo_cfg: PPOConfig,
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

        self.use_bc = ppo_cfg.use_bc
        self.target_kl = ppo_cfg.target_kl
        self.clip_vloss = ppo_cfg.clip_vloss
        ## network initialization

        self.online_lr = cfg.online_lr

        self.encoder = ActorCriticEncoder(cfg)
        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=3e-4)

        self.actor = Actorlog(cfg, self.encoder.repr_dim, use_std_share_network=ppo_cfg.use_std_share_network)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)

        # set old actor
        self.old_actor = deepcopy(self.actor)
        self.old_actor.requires_grad_(False)

        self.adv_compute_mode = ppo_cfg.adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae','gae2']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, self.encoder.repr_dim,)
            self.value_optim = optim.Adam(self.value.parameters(), lr=3e-4)
            if self.adv_compute_mode == 'q-v':
                self.doubleq = DoubleQMLP(cfg, self.encoder.repr_dim, )
                self.doubleq_optim = optim.Adam(self.doubleq.parameters(), lr=1e-4)
        elif self.adv_compute_mode in ['iql', 'iql2gae']:
            self.critic = IQLCritic(cfg, self.encoder.repr_dim,) 
        else:
            raise NotImplementedError

        self.ac_type = "ppo"
        self.fix_encoder = False
    
    @torch.no_grad()
    def predict_act(self, obs: torch.Tensor, eval_mode=False, **kwargss) -> ActorCriticOutput:
        assert obs.ndim == 4  # Ensure observation shape is correct
        h = self.encoder(obs).flatten(start_dim=1)
        if self.adv_compute_mode in ['tradition', 'gae2','gae'] and not eval_mode:
            # When tradition, return both action and state_value
            return *self.actor.get_action(h, eval_mode=eval_mode), self.value(h)
        return self.actor.get_action(h, eval_mode=eval_mode)
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.ndim == 4
        if self.fix_encoder:
            with torch.no_grad():
                h = self.encoder(obs).flatten(start_dim=1)
        else:
            h = self.encoder(obs).flatten(start_dim=1)
        return self.value(h)

    def bc_actor_update(self, rb: OfflineReplaybuffer):

        c = self.loss_cfg
        bacth = rb.sample(mini_batch_size=256)

        obs, act, reward, done, returns, obs_ = to_torch(bacth, self.device)
        # save obs as images
        # obs = obs.permute(0, 2, 3, 1)
        # obs = obs.cpu().numpy()
        # save_path = '/data0/jzn/workspace/Off2On-Visual/train_img.png'
        # from PIL import Image
        # img = Image.fromarray((obs[0]).astype('uint8'))
        # img.save(save_path)

        h = self.encoder(obs).flatten(start_dim=1)

        dist = self.actor.get_dist(h)
        log_prob = dist.log_prob(act)
        bc_loss = -log_prob.sum(dim=-1).mean() 
        # bc_loss = c.weight_bc_loss * bc_loss

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
        
        if hasattr(self, 'value_optim'):
            del self.value_optim

        if self.adv_compute_mode in ['iql2gae']:
            self.value = deepcopy(self.critic._value)
            del self.critic

        # params_list = list(self.actor.parameters()) + list(self.value.parameters()) + list(self.encoder.parameters())
        # self.optim = optim.Adam(params_list, lr=self.online_lr)
        self.optim = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.online_lr},
            {'params': self.value.parameters(), 'lr': 1e-4},
            {'params': self.encoder.parameters(), 'lr': self.online_lr},
        ])

        self.fix_encoder = True

        self.set_old_policy()
        # self.encoder.requires_grad_(False)
    
    def update(self, rb: OnlineReplayBuffer, iter=None, max_iter=None):

        c = self.loss_cfg
    
        batch = rb.sample_all()
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_values, dws = to_torch(batch, self.device)

        batch_reward = rews.sum() 
        success_times = batch_reward - (-rb.capacity)

        num_steps = obss.shape[0]
        
        if self.fix_encoder:
            with torch.no_grad():
                obss = self.encoder(obss).flatten(start_dim=1)
                next_obss = self.encoder(next_obss).flatten(start_dim=1)
        else:
            obss = self.encoder(obss).flatten(start_dim=1)
            next_obss = self.encoder(next_obss).flatten(start_dim=1)

        if self.use_bc:
            bc_act = self.old_actor.get_action(obss, eval_mode=True)

        # old_log_probs = old_log_probs.squeeze(-1)
        old_state_values = old_state_values.squeeze(-1) if old_state_values is not None else None

        truncs = torch.tensor(torch.logical_and(dones, torch.logical_not(dws)), dtype=torch.float32).to(self.device)
        end = torch.tensor(torch.logical_or(dws, truncs), dtype=torch.float32).to(self.device)

        # returns = compute_returns(rews, 0.99, dones)
        # returns = returns.to(self.device)
        # returns = returns.squeeze(-1)
        if self.adv_compute_mode == 'tradition':
            returns = compute_lambda_returns(rews, end, truncs, old_state_values, 0.99, self.gae_lambda)
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
            elif self.adv_compute_mode in ['gae','iql2gae']:
                vs = self.value(obss)
                next_vs = self.value(next_obss)
                advantages = torch.zeros_like(rews).to(self.device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    nextnonterminal = 1.0 - dones[t]
                    if t == num_steps - 1:
                        next_value = next_vs[t]
                    else:
                        next_value = vs[t + 1]
                    # real_next_values = next_not_done * nextvalues + final_values[t]
                    # delta = rews[t] + 0.99 * real_next_values - old_state_values[t]
                    delta = rews[t] + 0.99 * next_value * nextnonterminal - vs[t]
                    advantages[t] = lastgaelam = delta + 0.99 * self.gae_lambda * nextnonterminal * lastgaelam
                v_target = advantages + vs
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
                # adv = []
                # gae = 0
                # deltas = rews + 0.99 * (1.0 - dws) * next_vs - vs
                # for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                #     gae = delta + 0.99 * self.gae_lambda * gae * (1.0 - d)
                #     adv.insert(0, gae)
                # advantages = torch.tensor(adv, dtype=torch.float32).view(-1,1).to(self.device)
                # v_target = advantages + vs
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            elif self.adv_compute_mode == 'iql':
                advantages = self.critic.get_advantage(obss, acts)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        batch_size = rb.size
        clipfracs = []

        value_losses = []

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                # if not self.use_old_policy:
                dist = self.actor.get_dist(obss[index])
                log_prob = dist.log_prob(acts[index])

                if self.adv_compute_mode in ["tradition", "q-v"]:
                    state_value = self.value(obss[index].detach())
                    value_loss = F.mse_loss(state_value.squeeze(), returns[index])
                    q_loss = torch.tensor(0.0)
                    if self.adv_compute_mode == 'q-v':
                        with torch.no_grad():
                            next_act = self.actor.get_action(next_obss[index], eval_mode=True)
                            target_q = rews[index] + (1-dones[index]) * 0.99 * torch.min(*self.doubleq(next_obss[index], next_act))
                        q1, q2 = self.doubleq(obss[index], acts[index])
                        q_loss = 0.5 * F.mse_loss(q1, target_q) + 0.5 * F.mse_loss(q2, target_q)
                elif self.adv_compute_mode in ['gae','iql2gae']:
                    state_value = self.value(obss[index])
                    if self.clip_vloss:
                        v_loss_unclipped = (state_value- v_target[index]) ** 2
                        v_clipped = vs[index] + torch.clamp(state_value - vs[index], -self.clip_param, self.clip_param)
                        v_loss_clipped = (v_clipped - v_target[index]) ** 2
                        value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        value_loss = F.mse_loss(state_value, v_target[index])
                    value_losses.append(value_loss.item())
                    q_loss = torch.tensor(0.0)
                elif self.adv_compute_mode == "iql":
                    value_loss = torch.tensor(0.0)
                    q_loss = torch.tensor(0.0)
                    self.critic.update(obss[index], acts[index], rews[index], 1-dones[index], next_obss[index])

                logratio = log_prob.sum(1) - old_log_probs[index].sum(1).detach()
                ratios = torch.exp(logratio)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > self.clip_param).float().mean().item()]
        
                if approx_kl > self.target_kl:
                    break
                
                entropy_loss = -c.weight_entropy_loss * dist.entropy().sum(1, keepdim=True)
                surrogate = -advantages[index] * ratios
                surrogate_clipped = -advantages[index] * torch.clamp(ratios,
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                
                if self.use_bc:
                    bc_loss = - c.weight_bc_loss * dist.log_prob(bc_act[index]).sum(dim=-1, keepdim=True)
                else:
                    bc_loss = torch.tensor(0.0)
                    # bc_act = self.bc.get_action(obss[index], eval_mode=True)
                
                policy_loss = torch.max(surrogate, surrogate_clipped) + entropy_loss + bc_loss
                policy_loss = policy_loss.mean()

                if self.adv_compute_mode != 'iql':
                    critic_loss = value_loss + q_loss
                else:
                    self.critic.update(obss[index], acts[index], rews[index], 1-dones[index], next_obss[index])
        
                loss = policy_loss + critic_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.optim.step()

            if approx_kl > self.target_kl:
                break

        # if self.ppo_cfg.use_lr_decay and iter is not None and max_iter is not None:
        #     self.lr_decay(iter, max_iter)
        import pdb; pdb.set_trace()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.mean().item(),
                "surrogate": surrogate.mean().item(), "surrogate_clipped": surrogate_clipped.mean().item(),
                "bc_loss": bc_loss.mean().item(),
                "approx_kl": approx_kl.item(),
                "clipfracs": np.mean(clipfracs),
                "batch_reward": batch_reward.item(), "success_times": success_times.item(),} 
    
    def update_vector(self, rb: OnlineReplayBuffer,):

        c = self.loss_cfg
    
        batch = rb.sample_all()
        obss, acts, rews, next_obss, dones, old_log_probs, old_state_value = to_torch(batch, self.device)
        # obs shape
        # obss: (num_step, num_envs, 3, 128, 128)

        batch_reward = rews.sum() 
        success_times = batch_reward - (-rb.capacity)

        num_steps = obss.shape[0]
    
        with torch.no_grad():
            advantages = torch.zeros_like(rews).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - dones[t]
                if t == num_steps - 1:
                    next_value = self.get_value(next_obss[t]).reshape(1, -1)
                else:
                    next_value = old_state_value[t + 1]

                real_next_values = nextnonterminal * next_value
                # real_next_values = nextnonterminal * next_value + final_values[t] # t instead of t+1

                delta = rews[t] + 0.99 * real_next_values - old_state_value[t]
                advantages[t] = lastgaelam = delta + 0.99 * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + old_state_value
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        batch_size = rb.capacity
        clipfracs = []
        value_losses = []

        # flatten the batch
        b_obs = obss.reshape((-1,) + obss.shape[-3:])
        if self.fix_encoder:
            with torch.no_grad():
                b_obs = self.encoder(b_obs).flatten(start_dim=1)
        else:
            b_obs = self.encoder(b_obs).flatten(start_dim=1)

        b_logprobs = old_log_probs.reshape((-1,) + acts.shape[-1:])
        b_actions = acts.reshape((-1,) + acts.shape[-1:])
        b_advantages = advantages.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-10)
        b_returns = returns.reshape(-1)
        b_values = old_state_value.reshape(-1)

        for i in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                dist = self.actor.get_dist(b_obs[index])
                log_prob = dist.log_prob(b_actions[index])

                state_value = self.value(b_obs[index]).view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (state_value- b_returns[index]) ** 2
                    v_clipped = b_values[index] + torch.clamp(state_value - b_values[index], -self.clip_param, self.clip_param)
                    v_loss_clipped = (v_clipped - b_returns[index]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = F.mse_loss(state_value, b_returns[index])
                value_losses.append(value_loss.item())

                logratio = log_prob.sum(1) - b_logprobs[index].sum(1).detach()
                ratios = torch.exp(logratio)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratios - 1) - logratio).mean()
                    clipfracs += [((ratios - 1.0).abs() > self.clip_param).float().mean().item()]
        
                if approx_kl > self.target_kl:
                    break
                
                entropy_loss = -c.weight_entropy_loss * dist.entropy().sum(1, keepdim=True)
                surrogate = -b_advantages[index] * ratios
                surrogate_clipped = -b_advantages[index] * torch.clamp(ratios,
                                                       1.0 - self.clip_param, 1.0 + self.clip_param)
                
                policy_loss = torch.max(surrogate, surrogate_clipped) + entropy_loss
                policy_loss = policy_loss.mean()

                loss = policy_loss + 0.5 * value_loss

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.optim.step()

            if approx_kl > self.target_kl:
                break

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss, "entropy_loss": entropy_loss.mean().item(),
                "batch_reward": batch_reward.item(), "success_times": success_times.item(),}
    
    def update_multienv(self, rb: OnlineReplayBuffer, ):

        c = self.loss_cfg
    
        batch = rb.sample_all()
        obss, acts, rews, dones, old_log_probs, old_state_values, final_values, next_value,  = to_torch(batch, self.device)
        # obss, acts, rews, dones, old_log_probs, old_state_values, final_values, next_value, next_done = to_torch(batch, self.device)
        
        assert obss.shape[0] == acts.shape[0] == rews.shape[0] == dones.shape[0] == old_log_probs.shape[0] == old_state_values.shape[0]
        num_steps = obss.shape[0]

        batch_reward = rews.sum() 
        success_times = batch_reward - (-rb.capacity)
        
        value_losses = []

        with torch.no_grad():
            advantages = torch.zeros_like(rews).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                next_not_done = 1.0 - dones[t]
                if t == num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = old_state_values[t + 1]

                # if t == num_steps - 1:
                #     next_not_done = 1.0 - next_done
                #     nextvalues = next_value
                # else:
                #     next_not_done = 1.0 - dones[t + 1]
                #     nextvalues = old_state_values[t + 1]
    
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0, real_next_values=final_values
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
               
                delta = rews[t] + 0.99 * real_next_values - old_state_values[t]
                advantages[t] = lastgaelam = delta + 0.99 * self.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + old_state_values

        # flatten the batch
        b_obs = obss.reshape((-1,) + (3,128,128))
        if self.fix_encoder:
            with torch.no_grad():
                b_obs = self.encoder(b_obs).flatten(start_dim=1)
        else:
            b_obs = self.encoder(b_obs).flatten(start_dim=1)

        b_logprobs = old_log_probs.reshape((-1,) + (8,))
        b_actions = acts.reshape((-1,) + (8,))
        b_advantages = advantages.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-10)
        b_returns = returns.reshape(-1)
        b_values = old_state_values.reshape(-1)

        b_inds = np.arange(rb.capacity)
        clipfracs = []

        batch_size = rb.capacity
        mini_batch_size = self.mini_batch_size

        for epoch in range(self.K_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]

                # _, newlogprob, newvalue = self.predict_act(b_obs[mb_inds], )
                # logratio = newlogprob - b_logprobs[mb_inds]
                # ratio = logratio.exp()
                dist = self.actor.get_dist(b_obs[mb_inds])
                newlogprob = dist.log_prob(b_actions[mb_inds])
                logratio = newlogprob.sum(1) - b_logprobs[mb_inds].sum(1).detach()
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_param).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                if approx_kl > self.target_kl:
                    break

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = self.value(b_obs[mb_inds])
                newvalue = newvalue.view(-1)

                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_param, self.clip_param)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                value_losses.append(v_loss.item())

                loss = pg_loss + v_loss * 0.5

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
                self.optim.step()

            if approx_kl > self.target_kl:
                break

        return {"policy_loss": pg_loss.item(), "value_loss": v_loss,
                "approx_kl": approx_kl.item(), 
                "clipfracs": np.mean(clipfracs),
                "batch_reward": batch_reward.item(), 
                "success_times": success_times.item(),} 
    

    
    def set_old_policy(self):
        # if self.use_bc:
        #     pass
        # else:
        self.old_actor = deepcopy(self.actor)
        self.old_actor.requires_grad_(False)

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

        self.actor = Actorlog(cfg, state_dim, use_std_share_network=ppo_cfg.use_std_share_network)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.adv_compute_mode = ppo_cfg.adv_compute_mode

        if self.adv_compute_mode in ['tradition', 'q-v', 'gae']:
            # use GAE to estimate the advantage
            self.value = ValueMLP(cfg, state_dim,)
            self.value_optim = optim.Adam(self.value.parameters(), lr=3e-4)
            if self.adv_compute_mode == 'q-v':
                self.doubleq = DoubleQMLP(cfg, state_dim, )
                self.doubleq_optim = optim.Adam(self.doubleq.parameters(), lr=3e-4)
        elif self.adv_compute_mode == 'iql':
            self.critic = IQLCritic(cfg, state_dim,) 
        else:
            raise NotImplementedError

        self.ac_type = "ppo"
        
    @torch.no_grad()
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