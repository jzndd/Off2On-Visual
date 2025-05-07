import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple, Union
from rl_agent.baseagent import ActorCriticConfig
from rl_agent import utils
from torch import Tensor
from torch.distributions import Beta, Normal  
from rl_agent.cm import ConsistencyModel 

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
        self, cfg: ActorCriticConfig, repr_dim, use_ln: bool = False,
    ) -> None:
        super().__init__()
        self.use_trunk = cfg.use_trunk
        if self.use_trunk:
            self.input_layer = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.LayerNorm(cfg.feature_dim) if cfg.trunk_ln else nn.Identity(),
                nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh(),
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

    def compute_q(
        self, s: torch.Tensor, a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = self(s, a)
        return torch.min(q1, q2)
    
    def compute_loss(
        self, s: torch.Tensor, a: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = self(s, a)
        loss1 = F.mse_loss(q1, y)
        loss2 = F.mse_loss(q2, y)
        return loss1 + loss2
    
class ValueMLP(nn.Module):
    _net: torch.nn.modules.container.Sequential

    def __init__(
        self, cfg: ActorCriticConfig, repr_dim, use_ln: bool = False,
    ) -> None:
        super().__init__()
        self.use_trunk = cfg.use_trunk
        if self.use_trunk:
            self.input_layer = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.LayerNorm(cfg.feature_dim) if cfg.trunk_ln else nn.Identity(),
                nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh(),
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
        cfg: ActorCriticConfig,
        repr_dim: int,
        Q_lr: float = 1e-4,
        target_update_freq: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        v_lr: float = 1e-4,
        omega: float = 0.7,
    ) -> None:
        
        super().__init__()
        state_dim = repr_dim
        self._omega = omega

        # init double Q
        self._Q = DoubleQMLP(cfg, state_dim,)
        self._target_Q = deepcopy(self._Q)

        #for v
        self._value = ValueMLP(cfg, state_dim,)
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
        
class EnsembledLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 ensemble_size: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 0
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)

        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight + self.bias
        return out



class EnsembleCritic(nn.Module):
    '''
        Although SAC-RND is an ensemble-free method, this class
        is realised for convenience in using 2 separate Critics
        in a TD3 manner: 
            - https://arxiv.org/abs/1802.09477
            - https://arxiv.org/abs/2106.06860
    '''
    def __init__(self,
                 cfg: ActorCriticConfig,
                 repr_dim: int,
                 num_critics: int = 2,
                 edac_init: bool = True,) -> None:
        super().__init__()

        self.use_trunk = cfg.use_trunk
        self.repr_dim = repr_dim
        self.action_dim = cfg.num_actions
        self.hidden_dim = cfg.hidden_dim

        layer_norm = cfg.critic_ln

        if self.use_trunk:
            self.input_layer = nn.Sequential(
                nn.Linear(repr_dim, cfg.feature_dim),
                nn.LayerNorm(cfg.feature_dim) if cfg.trunk_ln else nn.Identity(),
                nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh(),
            )
            input_dim = cfg.feature_dim
        else:
            input_dim = repr_dim

        self.num_critics = num_critics

        self.critic = nn.Sequential(
            EnsembledLinear(input_dim + self.action_dim, cfg.hidden_dim, num_critics),
            nn.LayerNorm(cfg.hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(cfg.hidden_dim, cfg.hidden_dim, num_critics),
            nn.LayerNorm(cfg.hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(cfg.hidden_dim, cfg.hidden_dim, num_critics),
            nn.LayerNorm(cfg.hidden_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            EnsembledLinear(cfg.hidden_dim, 1, num_critics)
        )

        if edac_init:
            # init as in the EDAC paper
            for layer in self.critic[::3]:
                nn.init.constant_(layer.bias, 0.1)

            nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.use_trunk:
            h = self.input_layer(state)
            state = h
        concat = torch.cat([state, action], dim=-1)
        concat = concat.unsqueeze(0)
        concat = concat.repeat_interleave(self.num_critics, dim=0)
        q_values = self.critic(concat)
        return q_values
    
    def _reduce_ensemble(
        self, y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
    ) -> torch.Tensor:
        if reduction == "min":
            return torch.min(y, dim=dim)[0]
        elif reduction == "max":
            return torch.max(y, dim=dim)[0]
        elif reduction == "mean":
            return torch.mean(y, dim=dim)[0]
        elif reduction == "std":
            return torch.std(y, dim=dim)[0]
        elif reduction == "none":
            return y
        raise ValueError
    
    def compute_q(self, state, action, reduction="min") -> torch.Tensor: 
        Q = self(state,action)                     
        return self._reduce_ensemble(Q, reduction=reduction)
    
    def compute_q_weighted(
            self,
            state: torch.Tensor,
            action: Optional[torch.Tensor],
            reduction: str = "min",
            lam: float = 0.75,
        ) -> torch.Tensor:

        Q = self(state, action)
        Q = Q.unsqueeze(1)

        tuple_list = list(itertools.combinations(Q, 2))
        q_list = []
        for i in range(len(tuple_list)):
            values = torch.cat([tuple_list[i][0], tuple_list[i][1]])
            q_list.append(self._reduce_ensemble(values, reduction, lam=lam))

        return torch.stack(q_list, dim=0).mean(dim=0)
    
    def compute_loss(self, state, action, y):
        
        total_loss = 0.0
        q_list = self(state, action)

        for q in q_list:
            loss = F.mse_loss(q, y)
            total_loss += loss.mean()

        return total_loss
    
# ---------------------------- Actor ---------------------------- #

class VRL3Actor(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, repr_dim, ) -> None:
        super().__init__()

        # default params
        action_dim = cfg.num_actions
        feature_dim = cfg.feature_dim
        hidden_dim = cfg.hidden_dim
        self.use_trunk = cfg.use_trunk

        if self.use_trunk:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim) if cfg.trunk_ln else nn.Identity(),
                                    nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh())
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
    
class DrQv2Actor(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, repr_dim, use_std_share_network=False,) -> None:
        super().__init__()

        self.use_trunk = cfg.use_trunk
        if self.use_trunk:  
            self.trunk = nn.Sequential(nn.Linear(repr_dim, cfg.feature_dim),
                                    nn.LayerNorm(cfg.feature_dim) if cfg.trunk_ln else nn.Identity(),
                                    nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh(),)
            input_dim = cfg.feature_dim
        else:
            input_dim = repr_dim

        self.use_std_share_network = use_std_share_network
        if self.use_std_share_network:
            self.actor_linear = MLP(input_dim, cfg.hidden_dim, cfg.depth, cfg.num_actions * 2, activation=cfg.acitive_fn, final_activation=None, last_gain=0.01)
        else:
            self.actor_linear = MLP(input_dim, cfg.hidden_dim, cfg.depth, cfg.num_actions, activation=cfg.acitive_fn, final_activation=None, last_gain=0.01)
            self.log_std = nn.Parameter(torch.zeros(1, cfg.num_actions))

    def get_action(self, x: Tensor, std=None, eval_mode=False) -> Tensor:
        mean, std = self.forward(x, std)
        if eval_mode:
            action: Tensor = mean
            action = action.clamp(-1, 1)
            return action.detach()
        else:
            dist = utils.TruncatedNormal(mean, std)
            action: Tensor = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action)  # Summing over action dimensions
            return action, log_prob
        
    def get_dist(self, x: Tensor, std=None):
        mean, std = self.forward(x, std)
        return utils.TruncatedNormal(mean, std)

    def forward(self, x: Tensor, std=None):
        if self.use_trunk:
            x = self.trunk(x)

        if self.use_std_share_network:
            output = self.actor_linear(x)  # mean and std
            mean, log_std = output.split(output.shape[1] // 2, dim=1)
            std = torch.exp(log_std)
        else:
            mean = self.actor_linear(x)
            if std is None:
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
            else:
                std = torch.ones_like(mean) * std

        # deal with mean
        mean = torch.tanh(mean)
        
        return mean, std

# Actor Network
class CPActor(nn.Module):
    """
    A network module designed to function as an actor in a reinforcement learning framework,
    adhering to a specified consistency policy (CP). This actor network outputs action values
    given state inputs by utilizing an internal consistency model.

    Parameters:
    - repr_dim (int): Dimensionality of the state representation input to the model.
    - action_dim (int): Dimensionality of the action space.
    - device (str, optional): The device (e.g., 'cuda:1') on which the model computations will be performed.

    Output:
    - forward(state, return_dict=False): Processes the input state through the ConsistencyModel.
    """

    def __init__(self, cfg: ActorCriticConfig, repr_dim, device="cuda"):

        super(CPActor, self).__init__()

        self.device = device

        self.cm = ConsistencyModel(cfg, state_dim=repr_dim, device=device,)
        self.to(device)

    def forward(self, state):
        return self.cm(state)
    
    def to(self, device):
        super(CPActor, self).to(device)
    
    def loss(self, action, state):
        return self.cm.consistency_losses(action, state)
    
class Actorlog(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, repr_dim, use_std_share_network=False,) -> None:
        super().__init__()

        self.use_trunk = cfg.use_trunk
        if self.use_trunk:  
            self.trunk = nn.Sequential(nn.Linear(repr_dim, cfg.feature_dim),
                                    nn.LayerNorm(cfg.feature_dim) if cfg.trunk_ln else nn.Identity(),
                                    nn.ReLU() if cfg.trunk_activation == 'relu' else nn.Tanh(),)
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
        
    def get_dist(self, x: Tensor, std=None):
        mean, std = self.forward(x, std)
        return Normal(mean, std)

    def forward(self, x: Tensor, std=None):
        if self.use_trunk:
            x = self.trunk(x)

        if self.use_std_share_network:
            output = self.actor_linear(x)  # mean and std
            mean, log_std = output.split(output.shape[1] // 2, dim=1)
            std = torch.exp(log_std)
        else:
            mean = self.actor_linear(x)
            if std is None:
                log_std = self.log_std.expand_as(mean)
                std = torch.exp(log_std)
            else:
                std = torch.ones_like(mean) * std

        # deal with mean
        mean = torch.tanh(mean)
        
        return mean, std

# ---------------------------- Encoder ---------------------------- #
class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()

        # self.repr_dim = 32 * 57 * 57
        if cfg.img_size == 84:
            # 84 * 84 * 3 input
            self.repr_dim = 32 * 35 * 35
        elif cfg.img_size == 96:
            # 96 * 96 * 3 input
            self.repr_dim = 32 * 41 * 41
        elif cfg.img_size == 128:
            # 128 * 128 * 3 input
            self.repr_dim = 32 * 57 * 57
        else:
            raise ValueError("img_size should be 84, 96 or 128")

        self.convnet = nn.Sequential(nn.Conv2d(3 * cfg.frame_stack, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        # self.apply(utils.weight_init)

    def forward(self, obs):
        if obs.max() > 1. + 1e-5:
            # import pdb; pdb.set_trace()
            obs = obs / 255.0 * 2 - 1
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h