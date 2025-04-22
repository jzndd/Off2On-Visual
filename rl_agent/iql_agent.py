import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.rl_agent.utils import update_exponential_moving_average, to_torch

from models.rl_agent.net import DoubleQMLP, ValueMLP
from models.rl_agent.baseagent import ActorCriticConfig, BaseAgent
from replay_buffer import OfflineReplaybuffer, OnpolicyReplayBuffer

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(BaseAgent):
    def __init__(self, 
                 cfg: ActorCriticConfig, 
                 tau = 0.7, beta = 3.0, discount=0.99):
        super().__init__(cfg)

        # actor, encoder is already init in baseagent
        # default
        alpha = 0.005
        depth = 3
        action_dim = 4
        self.ac_type = "iql"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qf = DoubleQMLP(state_dim=self.encoder.repr_dim, 
                             feature_dim=cfg.feature_dim,
                             action_dim=action_dim, 
                             hidden_dim=cfg.hidden_dim, 
                             depth=depth)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False)
        self.vf = ValueMLP(state_dim=self.encoder.repr_dim,
                           feature_dim=cfg.feature_dim,
                           hidden_dim=cfg.hidden_dim,
                           depth=depth)

        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=1e-4)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=1e-4)
        self.actor_optimizer = torch.optim.Adam(self.qf.parameters(), lr=1e-4)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)

        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, 4e5)
        self.tau = tau               # iql params, for asymmetric_l2_loss
        self.beta = beta             # update policy
        self.discount = discount     # gamma
        self.alpha = alpha

        self.fix_encoder = False
        self.stage = 1

    def predict_act(self, obs: torch.Tensor, eval_mode=False, step=None):
        assert obs.ndim == 4  # Ensure observation shape is correct
        x = self.encoder(obs).flatten(start_dim=1)
        return self.actor.get_action(x, eval_mode=True)
        
    # indeed offline update, however, to keep consistency with other agents
    # we still use the func name "bc_update"
    # def bc_update(self, observations, actions, next_observations, rewards, terminals):
    def transfer_off2on(self):
        self.fix_encoder = True
        self.stage = 2


    def update(self, rb: Optional[Union[OfflineReplaybuffer,OnpolicyReplayBuffer]], step=None):   
        
        if self.stage == 1:
            batch = rb.sample(mini_batch_size=256)
            observations, actions, rewards, dones, _, next_observations  = to_torch(batch, device=self.device)
        else:
            batch = rb.sample(mini_batch_size=256)
            observations, actions ,rewards, next_observations, dones, _, _ = to_torch(batch, device=self.device)

        x = self.encoder(observations).flatten(start_dim=1)
        next_x = self.encoder(next_observations).flatten(start_dim=1)

        with torch.no_grad():
            target_q = torch.min(*self.q_target(x, actions))
            next_v = self.vf(next_x)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])
        if not self.fix_encoder:
            self.encoder_optimizer.zero_grad(set_to_none=True)

        # Update value function
        v = self.vf(x)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward(retain_graph=True)
        self.v_optimizer.step()
        

        # Update Q function
        # x = x.detach()
        targets = rewards + (1. - dones.float()) * self.discount * next_v.detach()
        qs = self.qf(x, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        # self.encoder_optimizer.zero_grad(set_to_none=True)
        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()
        # self.encoder_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        dist = self.actor.get_dist(x)
        bc_losses = -dist.log_prob(actions)
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.actor_optimizer.zero_grad(set_to_none=True)
        # self.encoder_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        # self.encoder_optimizer.step()
        if not self.fix_encoder:
            self.encoder_optimizer.step()

        return {
            'v_loss': v_loss.item(),
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
        }