import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

# X then E, both same shape
class LinearV2(nn.Module):
    __constants__ = ['in_features', 'out_features']
    size: int
    prev_size: Optional[int]
    weight_td: Optional[Tensor]
    weight_bu: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 size: int,
                 next_size: int = None,
                 bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 actv_mode: str = 'Wf(x)', # irrelevant, just for compatibility
                 gamma: float = 0.1,
                 beta: float = 1.0,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearV2, self).__init__()

        self.size = size
        self.next_size = next_size
        self.symmetric = symmetric
        self.actv_fn = actv_fn
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
        
        if next_size is not None:
            self.weight_td = Parameter(torch.empty((size, next_size), **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(size, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            if not symmetric:
                self.weight_bu = Parameter(torch.empty((next_size, size), **factory_kwargs))
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
        assert mode in ['zeros', 'rand', 'randn'], f"Invalid mode {mode}"
        if mode == 'zeros':
            state = {
                'x': torch.zeros((batch_size, self.size), device=self.device),
                'e': torch.zeros((batch_size, self.size), device=self.device),
            }
        elif mode == 'rand':
            state = {
                'x': torch.rand((batch_size, self.size), device=self.device) * 0.1,
                'e': torch.rand((batch_size, self.size), device=self.device) * 0.1,
            }
        elif mode == 'randn':
            state = {
                'x': torch.randn((batch_size, self.size), device=self.device) * 0.1,
                'e': torch.randn((batch_size, self.size), device=self.device) * 0.1,
            }
        return state

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, f_x_lp1):
        return F.linear(f_x_lp1, self.weight_td, self.bias)

    def update_e(self, state, f_x_lp1=None):
        if f_x_lp1 is not None:
            state['pred'] = self.predict(f_x_lp1)
        else:
            state['pred'] = state['x']
        state['e'] = state['x'] - state['pred']
        return state
    
    def update_x(self, state, bu_error=None):
        if bu_error is not None:
            state['x'] += self.gamma * (bu_error * self.d_actv_fn(state['x']))
        state['x'] += self.gamma * (-state['e'])
        return state
    
    def forward_error(self, state):
        error_prop = None
        if self.next_size is not None:
            weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
            error_prop = F.linear(state['e'], weight_bu, None)
        return error_prop
    
    def forward(self, state, bu_error=None, f_x_lp1=None) -> Tensor:

        state = self.update_e(state, f_x_lp1)

        state = self.update_x(state, bu_error)

        error_prop = self.forward_error(state)

        return state, error_prop
