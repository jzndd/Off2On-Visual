import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from rl_agent.baseagent import ActorCriticConfig
from rl_agent import utils
from torch import Tensor
from torch.distributions import Beta, Normal   

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

# ---------------------------- MLP Network ---------------------------- #
def MLP(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    activation: str = 'relu',
    final_activation: str = None,
    last_gain: float = 1.0,
) -> torch.nn.modules.container.Sequential:

    if activation == 'tanh':
        act_f = nn.Tanh()
    elif activation == 'relu':
        act_f = nn.ReLU()

    layers = [nn.Linear(input_dim, hidden_dim), act_f]
    orthogonal_init(layers[0])
    for _ in range(depth -1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        orthogonal_init(layers[-1])
        layers.append(act_f)

    layers.append(nn.Linear(hidden_dim, output_dim))
    orthogonal_init(layers[-1], gain=last_gain)
    if final_activation == 'relu':
        layers.append(nn.ReLU())
    elif final_activation == 'tanh':
        layers.append(nn.Tanh())
    else:
        layers = layers

    return nn.Sequential(*layers)



# ------------------------------------------- Critic network ______________________________
class DoubleQMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, cfg: ActorCriticConfig, repr_dim, use_ln: bool = False, use_trunk: bool = True
    ) -> None:
        super().__init__()
        self.use_trunk = use_trunk
        if self.use_trunk:
            self.input_layer = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.ReLU()
            )
            input_dim = cfg.feature_dim
        else:
            input_dim = repr_dim
        self._net1 = MLP(input_dim + cfg.num_actions, cfg.hidden_dim, cfg.depth, 1)
        self._net2 = MLP(input_dim + cfg.num_actions, cfg.hidden_dim, cfg.depth, 1)

    def forward(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_trunk:
            h = self.input_layer(s)
            sa = torch.cat([h, a], dim=1)
        else:
            sa = torch.cat([s, a], dim=1)
        return self._net1(sa), self._net2(sa)
    
class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, cfg: ActorCriticConfig, repr_dim, use_ln: bool = False, use_trunk: bool = True
    ) -> None:
        super().__init__()
        self.use_trunk = use_trunk
        if self.use_trunk:
            self.input_layer = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.ReLU()
            )
            input_dim = cfg.feature_dim
        else:
            input_dim = repr_dim
        self._net = MLP(input_dim, cfg.hidden_dim, cfg.depth, 1, activation=cfg.acitive_fn, final_activation=None)

    def forward(
        self, s: torch.Tensor
    ) -> torch.Tensor:
        if self.use_trunk:
            h = self.input_layer(s)
            return self._net(h)
        else:
            return self._net(s)

class IQLCritic(nn.Module):
    def __init__(
        self,
        cfg,
        repr_dim: int,
        # state_dim: int,
        # feature_dim: int,
        # action_dim: int,
        Q_lr: float = 1e-4,
        target_update_freq: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        v_lr: float = 1e-4,
        omega: float = 0.7,
        use_trunk: bool = True,
    ) -> None:
        
        super().__init__()
        state_dim = repr_dim
        self._omega = omega

        # init double Q
        self._Q = DoubleQMLP(cfg, state_dim, use_trunk=use_trunk)
        self._target_Q = DoubleQMLP(cfg, state_dim, use_trunk=use_trunk)
        self._target_Q.load_state_dict(self._Q.state_dict())

        #for v
        self._value = ValueMLP(cfg, state_dim, use_trunk=use_trunk)
        self._total_update_step = 0
        self._target_update_freq = target_update_freq
        self._tau = tau
        self._gamma = gamma

        self._v_optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=v_lr,
            )
        self._q_optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=Q_lr,
            )
    
    def minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._Q(s, a)
        return torch.min(Q1, Q2)

    def target_minQ(self, s: torch.Tensor, a: torch.Tensor):
        Q1, Q2 = self._target_Q(s, a)
        return torch.min(Q1, Q2)
    
    def expectile_loss(self, loss: torch.Tensor)->torch.Tensor:
        weight = torch.where(loss > 0, self._omega, (1 - self._omega))
        return weight * (loss**2)
    
    def update(self, s, a, r, not_done, s_) -> float:
        
        #     s, a, r, s_p, not_done = nobs_features, nactions[:, self.n_obs_steps - 1], batch['reward'][:, self.n_obs_steps - 1], next_nobs_features, batch['not_done'][:, self.n_obs_steps - 1]
        # else:
        #     s, a, r, s_p, not_done = nobs_features, nactions, batch['reward'], next_nobs_features, batch['not_done']
        # s, a, r, s_p, _, not_done, _, _ = replay_buffer.sample(self._batch_size)
        # Compute value loss

        with torch.no_grad():
            self._target_Q.eval()
            target_q = self.target_minQ(s, a)
        value = self._value(s)
        value_loss = self.expectile_loss(target_q - value).mean()

        #update v
        self._v_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self._v_optimizer.step()

        # Compute critic loss
        with torch.no_grad():
            self._value.eval()
            next_v = self._value(s_)
            
        target_q = r + not_done * self._gamma * next_v

        current_q1, current_q2 = self._Q(s, a)
        q_loss = ((current_q1 - target_q)**2 + (current_q2 - target_q)**2).mean()

        #update q and target q
        self._q_optimizer.zero_grad()
        q_loss.backward()
        self._q_optimizer.step()

        if self._total_update_step % self._target_update_freq == 0:
            for param, target_param in zip(self._Q.parameters(), self._target_Q.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

        return q_loss.detach().cpu().numpy(), value_loss.detach().cpu().numpy()
        
    def get_advantage(self, s, a)->torch.Tensor:

        q = self.minQ(s, a)
        v = self._value(s)
        adv = q - v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv

    def transbc2online(self, online_lr: float = 3e-6):
        self._v_optimizer = torch.optim.Adam(
            self._value.parameters(), 
            lr=online_lr,
            )
        self._q_optimizer = torch.optim.Adam(
            self._Q.parameters(),
            lr=online_lr,
            )
    
# ---------------------------- Actor ---------------------------- #

class VRL3Actor(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, repr_dim, use_trunk=True) -> None:
        super().__init__()

        # default params
        action_dim = cfg.num_actions
        feature_dim = cfg.feature_dim
        hidden_dim = cfg.hidden_dim
        self.use_trunk = use_trunk

        if self.use_trunk:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim))
            input_dim = feature_dim
        else:
            input_dim = repr_dim

        self.actor_linear = MLP(input_dim, hidden_dim, cfg.depth, action_dim, final_activation=None)

        self.action_shift=0
        self.action_scale=1
        self.apply(utils.weight_init)

    def forward(self, obs, std):
        if self.use_trunk:
            h = self.trunk(obs)
        else:
            h = obs

        mu = self.actor_linear(h)
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def forward_with_pretanh(self, obs, std):

        if self.use_trunk:
            h = self.trunk(obs)
        else:
            h = obs

        mu = self.actor_linear(h)
        pretanh = mu
        mu = torch.tanh(mu)
        mu = mu * self.action_scale + self.action_shift
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist, pretanh
    
class Actorlog(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, repr_dim, use_trunk=True, use_std_share_network=False) -> None:
        super().__init__()

        self.use_trunk = use_trunk
        if self.use_trunk:  
            self.trunk = nn.Sequential(nn.Linear(repr_dim, cfg.feature_dim),
                                    nn.ReLU())
            input_dim = cfg.feature_dim
        else:
            input_dim = repr_dim

        self.use_std_share_network = use_std_share_network
        if self.use_std_share_network:
            self.actor_linear = MLP(input_dim, cfg.hidden_dim, cfg.depth, cfg.num_actions * 2, activation=cfg.acitive_fn, final_activation=None, last_gain=0.01)
        else:
            self.actor_linear = MLP(input_dim, cfg.hidden_dim, cfg.depth, cfg.num_actions, activation=cfg.acitive_fn, final_activation=None, last_gain=0.01)
            self.log_std = nn.Parameter(torch.zeros(1, cfg.num_actions))

    def get_action(self, x: Tensor, eval_mode=False) -> Tensor:
        mean, std = self.forward(x)
        if eval_mode:
            action: Tensor = mean
            action = action.clamp(-1, 1)
            return action.detach()
        else:
            dist = Normal(mean, std)
            action: Tensor = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action)  # Summing over action dimensions
            return action, log_prob
        
    def get_dist(self, x: Tensor):
        mean, std = self.forward(x)
        return Normal(mean, std)

    def forward(self, x: Tensor):
        if self.use_trunk:
            x = self.trunk(x)

        if self.use_std_share_network:
            output = self.actor_linear(x)  # mean and std
            mean, log_std = output.split(output.shape[1] // 2, dim=1)
        else:
            mean = self.actor_linear(x)
            log_std = self.log_std.expand_as(mean)

        # log_std_clip
        # log_std = log_std.clamp(-5, 0.)

        # deal with std
        std = torch.exp(log_std)

        # deal with mean
        mean = torch.tanh(mean)
        
        return mean, std

# ---------------------------- Encoder ---------------------------- #
class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()

        # self.repr_dim = 32 * 57 * 57
        # 84 * 84 * 3 input
        # self.repr_dim = cfg.frame_stack * 32 * 35 * 35

        # 96 * 96 * 3 input
        self.repr_dim = cfg.frame_stack * 32 * 41 * 41

        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        # self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h