import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

# Whittington & Bogacz 2017
class PrecisionWeighted(nn.Module):
    __constants__ = ['in_features', 'out_features']
    size: int
    prev_size: Optional[int]
    weight_td: Optional[Tensor]
    weight_var: Optional[Tensor]
    weight_bu: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 size: int,
                 prev_size: int = None,
                 bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 beta: float = 1.0,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PrecisionWeighted, self).__init__()

        self.size = size
        self.prev_size = prev_size
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
        
        self.weight_var = torch.empty((size, size), **factory_kwargs)
        if prev_size is not None:
            self.weight_td = Parameter(torch.empty((prev_size, size), **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(prev_size, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            if not symmetric:
                self.weight_bu = Parameter(torch.empty((size, prev_size), **factory_kwargs))
            else:
                self.register_parameter('weight_bu', None)
        else:
            self.register_parameter('weight_td', None)
            self.register_parameter('weight_bu', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_var, a=math.sqrt(5))
        self.weight_var.data += torch.eye(self.size, device=self.device)
        self.weight_var.data *= 0.1
        self.weight_var.data = torch.clamp(self.weight_var.data, min=0.001)
        if self.prev_size is not None:
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            self.weight_td.data *= 0.1
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            if not self.symmetric:
                nn.init.kaiming_uniform_(self.weight_bu, a=math.sqrt(5))
                self.weight_bu.data *= 0.1
            
    def init_state(self, batch_size):
        return {
            'x': torch.zeros((batch_size, self.size), device=self.device),
            'e': torch.zeros((batch_size, self.size), device=self.device),
            'eps': torch.zeros((batch_size, self.size), device=self.device)
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.weight_var = self.weight_var.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def predict(self, state):
        return F.linear(self.actv_fn(state['x']), self.weight_td, self.bias)
    
    def update_x(self, state, e_below=None):
        if e_below is not None:
            weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
            update = F.linear(e_below, weight_bu, None)
            state['x'] += self.gamma * (-state['e'] + update * self.d_actv_fn(state['x']))
        state['x'] += self.gamma * self.beta * (-state['e'])
        
    def update_e(self, state, pred=None):
        if pred is not None:
            state['pred'] = pred
            state['eps'] += (self.gamma * 5.0) * (F.linear(state['e'], self.weight_var, None) - state['eps'])
            # state['eps'] = F.linear(state['e'], self.weight_var, None)
        else:
            state['pred'] = state['x']
        
        state['e'] += (self.gamma * 5.0) * (state['x'] - state['pred'] - state['eps'])

    def update_grad(self, state, e_below):
        b_size = e_below.shape[0]
        self.weight_td.grad = -(e_below.T @ self.actv_fn(state['x'])) / b_size
        if self.bias is not None:
            self.bias.grad = -e_below.mean(dim=0)
        if not self.symmetric:
            self.weight_bu.grad = -(self.actv_fn(state['x']).T @ e_below) / b_size
        self.weight_var.grad = -(((state['eps'].T @ state['e']) / b_size) - torch.eye(state['e'].shape[1], device=self.device))