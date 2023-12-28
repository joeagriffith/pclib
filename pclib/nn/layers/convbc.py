import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import Optional, Tuple
from pclib.utils.functional import reTanh
from pclib.nn.layers import Conv2d

# Whittington & Bogacz 2017
class Conv2dBc(Conv2d):
    
    def __init__(self,
                 prev_shape: Optional[Tuple[int]],
                 shape: (int, int, int), # (channels, height, width)
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding = 1,
                 maxpool=1,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.tanh,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:
        
        # assert stride == 1, "Stride != 1 not yet supported."

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            prev_shape,
            shape,
            kernel_size,
            stride,
            padding,
            maxpool,
            has_bias,
            symmetric,
            actv_fn,
            d_actv_fn,
            gamma,
            device,
            dtype,
        )
        
    def update_x(self, state, e_below=None, pred=None, temp=None):
        state['x'] = state['x'].detach()
        if e_below is not None:
            if e_below.dim() == 2:
                e_below = e_below.unsqueeze(-1).unsqueeze(-1)
            update = self.propagate(e_below)
            state['x'] += self.gamma * (update * self.d_actv_fn(state['x']))
            if pred is not None:
                state['x'] += self.gamma * pred
            if temp is not None:
                eps = torch.randn_like(state['x'], device=self.device) * 0.034 * temp
                state['x'] += eps
            
            state['x'] = self.norm(state['x'])