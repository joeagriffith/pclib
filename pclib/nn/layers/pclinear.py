import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional


class PCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    weight_td: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 symmetric: bool = True,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric

        self.device = device

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if not symmetric:
            self.weight_td = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        else:
            self.register_parameter('weight_td', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data *= 0.1
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if not self.symmetric:
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            self.weight_td.data *= 0.1
            

    # Returns a tuple of two tensors, r and e, of shape (batch_size, out_features) and (batch_size, in_features) respectively
    def init_state(self, batch_size):
        state = [
            torch.zeros((batch_size, self.out_features), device=self.device),
            torch.zeros((batch_size, self.in_features), device=self.device)
        ]
        # if bias: # TODO: is r_bias a better initialisation than zeros?
            # r += self.bias
        return state

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x, state) -> Tensor:

        state[1] = x - F.linear(state[0], self.weight.T, self.bias)
        update = F.linear(state[1], self.weight, None) if self.symmetric else F.linear(state[1], self.weight_td.T, None)
        state[0] += update

        return state