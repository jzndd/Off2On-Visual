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
sys.path.append("/DATA/disk0/jzn/Diamond/src/")

from meta_world.action_processing import ACTION_DIM_POSSIBLES, decode_metaworld_action
from replay_buffer import OnpolicyReplayBuffer
from torch.distributions import Normal
from models.rl_agent.baseagent import *
from models.rl_agent.utils import compute_returns, compute_lambda_returns

class ReinforceAgent(BaseAgent):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__(cfg)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

        self.ac_type = "reinforce"

        self.bc_loss = nn.MSELoss()

    def predict_act_value(self, obs: Tensor, eval_mode=False) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        return ActorCriticOutput(self.actor.get_action(x, eval_mode), self.critic(x))

    def forward(self, rb: OnpolicyReplayBuffer, batch = None):
        c: ActorCriticLossConfig = self.loss_cfg
        
        if batch is not None:
            # convert batch.action(Da=84) to actual_act(Da=4)
            expert_act = decode_metaworld_action(batch.act.squeeze(1))  # B x Da (B, 4)
            expert_obs = batch.obs.squeeze(1)
            x = self.encoder(expert_obs)
            x = x.flatten(start_dim=1)
            pred_act = self.actor.get_action(x, is_resample=True)
            bc_loss = c.weight_bc_loss * self.bc_loss(pred_act, expert_act)
        else:
            raise ValueError("batch is None")

        # use reinforce to update the policy
        obss, acts, rews, next_obss, dones, truncs = rb.sample(mini_batch_size=rb.size)

        device = batch.obs.device
        obss = obss.to(device)
        acts = acts.to(device)
        rews = rews.to(device)
        next_obss = next_obss.to(device)
        dones = dones.to(device)
        truncs = truncs.to(device)
        
        x = self.encoder(obss)
        x = x.reshape(x.shape[0], -1)

        # critic
        val = self.critic(x.detach())
        returns = compute_lambda_returns(rews, dones, truncs, val, c.gamma, c.lambda_)
        critic_loss = F.mse_loss(val, returns)
        
        with torch.no_grad():
            advantages = (returns - val).detach()
            # advantages = returns - val.detach()
        mean, std = self.actor(x)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(acts).sum(dim=1)
        entropy_loss = -c.weight_entropy_loss * dist.entropy().mean()
        improve_loss = -(log_probs * advantages).mean()
        policy_loss = improve_loss + entropy_loss + bc_loss

        metrics = {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "bc_loss": bc_loss.item(),
            "improve_loss": improve_loss.item()
        }

        return policy_loss, critic_loss, metrics
    
    def update(self, rb: OnpolicyReplayBuffer, batch = None):
        policy_loss, critic_loss, metrics = self.forward(rb, batch)
        print("policy loss is {}".format(policy_loss))
        print("critic loss is {}".format(critic_loss))
        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        policy_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        self.encoder_opt.step()
        self.actor_opt.step()
        self.critic_opt.step()
        return metrics


 