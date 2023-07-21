import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional


class PCLinearUni(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCLinearUni, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.device = device

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.weight.data *= 0.1

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
        update = F.linear(state[1], self.weight, None)
        state[0] += update

        return state

class PCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight_td: Tensor
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCLinearUni, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.device = device

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_td = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    # Returns a tuple of two tensors, r and e, of shape (batch_size, out_features) and (batch_size, in_features) respectively
    def init_state(self, batch_size, bias=False):
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

        state[1] = x - F.linear(state[0], self.weight_td, self.bias)
        state[0] += F.linear(state[1], self.weight, None)

        return state