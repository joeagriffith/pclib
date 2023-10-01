import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional


class Linear(nn.Module):
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
                 d_actv_fn: callable = None,
                 actv_mode: str = 'Wf(x)', # 'f(Wx)' or 'Wf(x)'
                 gamma: float = 0.1,
                 beta: float = 1.0,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.actv_mode = actv_mode
        self.gamma = gamma
        self.beta = beta
        self.device = device

        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()

        self.weight_td = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
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
        self.weight_td.data *= 0.1
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if not self.symmetric:
            nn.init.kaiming_uniform_(self.weight_bu, a=math.sqrt(5))
            self.weight_bu.data *= 0.1
            

    # Returns a tuple of two tensors, x and e, of shape (batch_size, out_features) and (batch_size, in_features) respectively
    def init_state(self, batch_size, mode='zeros'):
        assert mode in ['zeros', 'rand'], f"Invalid mode {mode}"
        if mode == 'zeros':
            state = {
                'x': torch.zeros((batch_size, self.out_features), device=self.device),
                'e': self.bias.detach().clone(),
            }
        elif mode == 'rand':
            state = {
                'x': torch.rand((batch_size, self.out_features), device=self.device) * 0.1,
                'e': self.bias.detach().clone(),
            }
        return state

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def forward(self, x_below, state, td_error=None) -> Tensor:

        weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
        
        if self.actv_mode == 'Wf(x)':
            state['pred'] = F.linear(self.actv_fn(state['x']), self.weight_td, self.bias)
            state['e'] = x_below - state['pred']
            update = F.linear(state['e'], weight_bu, None) * self.d_actv_fn(state['x'])
        elif self.actv_mode == 'f(Wx)':
            state['pred'] = F.linear(state['x'], self.weight_td, self.bias)
            state['e'] = x_below - self.actv_fn(state['pred'])
            update = F.linear(state['e'] * self.d_actv_fn(state['pred']), weight_bu, None)
        else:
            raise ValueError(f"Invalid actv_mode {self.actv_mode}, must be one of ['f(Wx)', 'Wf(x)']")

        if td_error is not None:
            update -= self.beta * td_error

        state['x'] += self.gamma * update

        return state
