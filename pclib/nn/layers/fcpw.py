import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from typing import Optional
import math

from pclib.nn.layers import FC

class FCPW(FC):
    """
    | Precision weighted fully connected layer with optional bias and optionally symmetric weights.    
    | Attempts to learn the variance of the prediction error, and uses this to weight the error signal.

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
        | weight_var (torch.Tensor): Weights for precision-weighting the error signal.

    """
    weight_var: Optional[Tensor]

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

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, has_bias, symmetric, actv_fn, d_actv_fn, gamma, **factory_kwargs)

        self.weight_var = torch.empty((out_features, out_features), **factory_kwargs)
        nn.init.kaiming_uniform_(self.weight_var, a=math.sqrt(5))
        self.weight_var.data += torch.eye(self.size, device=self.device)
        self.weight_var.data *= 0.1
        self.weight_var.data = torch.clamp(self.weight_var.data, min=0.001)
            
    def init_state(self, batch_size):
        """
        | Builds a new state dictionary for the layer.
        | Introduces 'eps' which is used to calculate raw prediction error, whilst 'e' is precision-weighted.

        Args:
            | batch_size (int): Batch size of the state.

        Returns:
            | state (dict): Dictionary containing 'x', 'e', and 'eps' tensors of shape (batch_size, out_features).
        """
        return {
            'x': torch.zeros((batch_size, self.size), device=self.device),
            'e': torch.zeros((batch_size, self.size), device=self.device),
            'eps': torch.zeros((batch_size, self.size), device=self.device)
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.weight_var = self.weight_var.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def update_e(self, state, pred=None):
        """
        | Calculates precision-weighted errors by implementing an iterative update rule for 'e' and 'eps'.
        | At the fixed point, and 'eps' is the prediction error, and 'e' is the precision-weighted prediction error.

        Args:
            | state (dict): Dictionary containing 'x', 'e', and 'eps' tensors for this layer.
            | pred (torch.Tensor): Top-down prediction of state['x'].
        """
        if pred is not None:
            if pred.dim() == 4:
                pred = pred.flatten(1)
            state['pred'] = pred
            state['eps'] += (self.gamma * 5.0) * (F.linear(state['e'], self.weight_var, None) - state['eps'])
            # state['eps'] = F.linear(state['e'], self.weight_var, None)
        else:
            state['pred'] = state['x']
        
        state['e'] += (self.gamma * 5.0) * (state['x'] - state['pred'] - state['eps'])

    def update_grad(self, state, e_below):
        """
        | Not implemented.
        """
        raise(NotImplementedError)
        # b_size = e_below.shape[0]
        # self.weight_td.grad = -(e_below.T @ self.actv_fn(state['x'])) / b_size
        # if self.bias is not None:
        #     self.bias.grad = -e_below.mean(dim=0)
        # if not self.symmetric:
        #     self.weight_bu.grad = -(self.actv_fn(state['x']).T @ e_below) / b_size
        # self.weight_var.grad = -(((state['eps'].T @ state['e']) / b_size) - torch.eye(state['e'].shape[1], device=self.device))