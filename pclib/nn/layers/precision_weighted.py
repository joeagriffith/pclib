import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional


class PrecisionWeighted(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight_td: Tensor
    weight_bu: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 gamma: float = 0.1,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PrecisionWeighted, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.gamma = gamma

        self.device = device

        self.weight_td = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.weight_var = Parameter(torch.empty((in_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if not symmetric:
            self.weight_bu = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        else:
            self.register_parameter('weight_bu', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
        # self.weight_td.data *= 0.1
        nn.init.kaiming_uniform_(self.weight_var, a=math.sqrt(5))
        # self.weight_var.data *= 0.1
        self.weight_var.data += torch.eye(self.weight_var.shape[0], device=self.device)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if not self.symmetric:
            nn.init.kaiming_uniform_(self.weight_bu, a=math.sqrt(5))
            # self.weight_bu.data *= 0.1
            

    # Returns a tuple of two tensors, r and e, of shape (batch_size, out_features) and (batch_size, in_features) respectively
    def init_state(self, batch_size, mode='zeros'):
        assert mode in ['zeros', 'rand', 'randn'], f"Invalid mode {mode}"
        if mode == 'zeros':
            state = [
                torch.zeros((batch_size, self.out_features), device=self.device),
                torch.zeros((batch_size, self.in_features), device=self.device),
                torch.zeros((batch_size, self.in_features), device=self.device)
            ]
        elif mode == 'rand':
            state = [
                torch.rand((batch_size, self.out_features), device=self.device) * 0.1,
                torch.rand((batch_size, self.in_features), device=self.device) * 0.1,
                torch.zeros((batch_size, self.in_features), device=self.device)
            ]
        elif mode == 'randn':
            state = [
                torch.randn((batch_size, self.out_features), device=self.device) * 0.1,
                torch.randn((batch_size, self.in_features), device=self.device) * 0.1,
                torch.zeros((batch_size, self.in_features), device=self.device)
            ]
        # if bias: # TODO: is r_bias a better initialisation than zeros?
            # r += self.bias
        return state

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x, state, td_error=None) -> Tensor:

        state[2] += self.gamma * (F.linear(state[1], self.weight_var, None) - state[2])

        pred = F.linear(self.actv_fn(state[0]), self.weight_td, self.bias)
        state[1] += self.gamma * (x - pred - state[2])

        weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
        assert self.actv_fn == F.relu, "Only relu supported for now"
        update = F.linear(state[1], weight_bu, None) * torch.sign(torch.relu(state[0]))
        if td_error is not None:
            update -= td_error
        state[0] += self.gamma * update

        return state
