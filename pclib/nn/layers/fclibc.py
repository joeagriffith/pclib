import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity
from pclib.nn.layers import FCLI

class FCLIBC(FCLI):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | The layer stores its state in a dictionary with keys 'x' and 'e'.
    | Layer is defined such that 'x' and 'e' are the same shape, and 'x' precedes 'e' in the architecture.
    | The Layer implements predictions as: Wf(x) + Optional(b).

    Args:
        | in_features (int): Number of input features.
        | out_features (int): Number of output features.
        | has_bias (bool): Whether to include a bias term.
        | symmetric (bool): Whether to reuse top-down prediction weights, for bottom-up error propagation.
        | actv_fn (callable): Activation function to use.
        | d_actv_fn (callable): Derivative of activation function to use (if None, will be inferred from actv_fn).
        | gamma (float): step size for x updates.
        | device (torch.device): Device to use for computation.
        | dtype (torch.dtype): Data type to use for computation.

    Attributes:
        | weight_td (torch.Tensor): Weights for top-down predictions.
        | weight_bu (torch.Tensor): Weights for bottom-up predictions (if symmetric=False).
        | bias (torch.Tensor): Bias term (if has_bias=True).
    """
    __constants__ = ['in_features', 'out_features']
    in_features: Optional[int]
    out_features: int
    weight_td: Optional[Tensor]
    weight_bu: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features,
                         out_features,
                         has_bias,
                         symmetric,
                         actv_fn,
                         d_actv_fn,
                         gamma,
                         device,
                         dtype,
                        )   

    def update_x(self, state, e_below=None, pred=None, temp=None):
        """
        | Calculates a new_x and then interpolates between the current state['x'] and new_x, updating state['x'] inplace.
        | This uses the lateral connectivity to produce a target value, rather than an incremental update.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below Optional([torch.Tensor]): Error of layer below. if None, no gradients are calculated.
        """
        state['x'] = (1.0 - self.gamma) * state['x'] + self.gamma * self.lateral(state)
        if e_below is not None:
            update = self.propagate(e_below)
            state['x'] += self.gamma * (update * self.d_actv_fn(state['x']))
        if pred is not None:
            state['x'] += self.gamma * pred

        if temp is not None:
            eps = torch.randn_like(state['x'].detach(), device=self.device) * temp * 0.034
            state['x'] += eps
        
        state['x'] = self.norm(state['x'])