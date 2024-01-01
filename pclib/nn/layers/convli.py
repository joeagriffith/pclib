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
class Conv2dLi(Conv2d):

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
            prev_shape=prev_shape,
            shape=shape,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            maxpool=maxpool,
            has_bias=has_bias,
            symmetric=symmetric,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            device=device,
            dtype=dtype
        )

        self.lateral_layer = nn.Conv2d(self.shape[0], self.shape[0], kernel_size=3, stride=1, padding=1, bias=False, **self.factory_kwargs)
        nn.init.dirac_(self.lateral_layer.weight)
        self.calc_mean = nn.Conv2d(self.shape[0], self.shape[0], kernel_size=3, stride=1, padding=1, bias=False, **self.factory_kwargs)
        # set weights to 1/num
        self.calc_mean.weight.data.fill_(1.0 / (kernel_size**2*self.shape[0]))

        self.moving_avg = torch.ones(self.shape, **self.factory_kwargs)
    
    def to(self, *args, **kwargs):
        self.device = args[0]
        self.moving_avg = self.moving_avg.to(self.device)
        return super().to(*args, **kwargs)

    def lateral(self, state):
        return self.lateral_layer(self.boost(state['x']))
    
    def update_x(self, state, e_below=None, temp=None):
        # If not input layer, propagate error from layer below
        state['x'] = (1.0 - self.gamma) * state['x'] + self.gamma * self.lateral(state['x'].detach())
        if e_below is not None:
            if e_below.dim() == 2:
                e_below = e_below.unsqueeze(-1).unsqueeze(-1)
            update = self.propagate(e_below)
            state['x'] += self.gamma * (update * self.d_actv_fn(state['x']))

        state['x'] += self.gamma * (-state['e'])
        
        if temp is not None:
            eps = torch.randn_like(state['x'], device=self.device) * 0.034 * temp
            state['x'] += eps
        
        state['x'] = self.norm(state['x'])

    def update_mov_avg(self, state):
        self.moving_avg = 0.99 * self.moving_avg + 0.01 * state['x'].detach().mean(0)

    def boost(self, x):
        # return x
        mult = self.moving_avg / self.calc_mean(self.moving_avg)
        return x * mult

    # def update_x(self, state, e_below=None, pred=None, temp=None):
    #     if e_below is not None:
    #         if e_below.dim() == 2:
    #             e_below = e_below.unsqueeze(-1).unsqueeze(-1)
    #         update = self.propagate(e_below)
    #         state['x'] += self.gamma * (update * self.d_actv_fn(state['x']))
    #         if pred is not None:
    #             state['x'] += self.gamma * pred
    #         if temp is not None:
    #             eps = torch.randn_like(state['x'], device=self.device) * 0.034 * temp
    #             state['x'] += eps
            
    #         state['x'] = self.norm(state['x'])

    def update_grad(self, state, e_below=None):
        """
        
        """
        if e_below is not None:
            b_size = e_below.shape[0]
            x = F.interpolate(self.actv_fn(state['x'].detach()), scale_factor=self.maxpool, mode='nearest')
            self.conv[0].weight.grad = 2*-conv2d_weight(e_below, self.conv[0].weight.shape, x, stride=self.stride, padding=self.padding, dilation=1, groups=1) / b_size
            # I dont trust this
            # Might need separate bias for prediction. this is for error prop
            if self.has_bias:
                self.bias.grad = 2*-e_below.mean(0)