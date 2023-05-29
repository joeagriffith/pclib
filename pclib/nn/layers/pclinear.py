import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

class PCLinearUniweighted(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 
                 nu=1.0,
                 mu=1.0,
                 eta=0.1,
                 relu_errs=True,

                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(PCLinearUniweighted, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.nu = nu
        self.mu = mu
        self.eta = eta
        self.relu_errs = relu_errs

        self.device = device

        self.weight = Parameter(torch.empty(out_features, in_features), **factory_kwargs)
        if bias:
            self.bias = Parameter(torch.empty(out_features), **factory_kwargs)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def init_vars(self, batch_size, bias=False):
        e = torch.zeros((batch_size, self.in_size), device=self.device)
        r = torch.zeros((batch_size, self.out_size), device=self.device)
        if bias:
            r += self.bias
        return e, r

    def _calc_err(self, input: Tensor, td_pred: Tensor) -> Tensor:
        e = input - td_pred
        if self.relu_errs:
            e = F.relu(e)
        return e

    def forward(self, input: Tensor, r: Tensor, td_err=None) -> Tensor:
        td_pred = F.linear(r, self.weight.t())
        e = self._calc_err(input, td_pred)

        update = F.linear(e, self.weight, self.bias)

        r = (self.nu * r) + (self.mu * update) + (self.eta * td_err)

        return r, e



        