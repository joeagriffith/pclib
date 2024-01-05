import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity
from pclib.nn.layers import FC

class FCDP(FC):
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

        super().__init__(
            in_features,
            out_features,
            has_bias,
            symmetric,
            actv_fn,
            d_actv_fn,
            gamma,
            device,
            dtype
        )

        self.init_params()
        self.out_sign_correct = torch.cat([torch.ones(self.out_features, device=self.device), -torch.ones(self.out_features, device=self.device)], dim=0)
        if self.in_features is not None:
            self.in_sign_correct = torch.cat([torch.ones(self.in_features, device=self.device), -torch.ones(self.in_features, device=self.device)], dim=0)

    def to(self, device):
        """
        | Moves layer to device and updates device attribute.

        Args:
            | device (torch.device): Device to move to.
        """
        super().to(device)
        self.out_sign_correct = self.out_sign_correct.to(device)
        if self.in_features is not None:
            self.in_sign_correct = self.in_sign_correct.to(device)
        
    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on init args.
        """
        if self.in_features is not None:
            self.weight_td = Parameter(torch.empty((self.in_features*2, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            # nn.init.kaiming_normal_(self.weight_td, a=math.sqrt(5))
            # nn.init.xavier_uniform_(self.weight_td)
            # nn.init.xavier_normal_(self.weight_td)
            

            if self.has_bias:
                self.bias = Parameter(torch.empty(self.in_features*2, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter('bias', None)

            if not self.symmetric:
                self.weight_bu = Parameter(torch.empty((self.out_features, self.in_features*2), **self.factory_kwargs))
                nn.init.kaiming_uniform_(self.weight_bu, a=math.sqrt(5))
            else:
                self.register_parameter('weight_bu', None)

        else:
            self.register_parameter('weight_td', None)
            self.register_parameter('weight_bu', None)
            self.register_parameter('bias', None)
            
    def init_state(self, batch_size):
        """
        | Builds a new state dictionary for the layer.

        Args:
            | batch_size (int): Batch size of the state.

        Returns:
            | state (dict): Dictionary containing 'x' and 'e' tensors of shape (batch_size, out_features).

        """
        return {
            'x': torch.zeros((batch_size, self.out_features), device=self.device),
            'e': torch.zeros((batch_size, self.out_features*2), device=self.device),
        }
    
    # Recalculates prediction-error (state['e']) between state['x'] and a top-down prediction of it
    # With simulated annealing
    def update_e(self, state, pred=None, temp=None):
        """
        | Updates prediction-error (state['e']) inplace between state['x'] and the top-down prediction of it.
        | Uses simulated annealing if temp is not None.
        | Does nothing if pred is None. This is useful so the output layer doesn't need specific handling.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | pred (Optional[torch.Tensor]): Top-down prediction of state['x'].
            | temp (Optional[float]): Temperature for simulated annealing.
        """
        if pred is not None:
            if pred.dim() == 4:
                pred = pred.flatten(1)
            x = torch.cat([state['x'].detach(), state['x'].detach()], dim=1)
            state['e'] = (x - pred) * self.out_sign_correct

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * temp * 0.034
            state['e'] += eps
    
    def update_x(self, state, e_below=None, d_pred=None, temp=None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x)).

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. None if input layer.
        """
        # If not input layer, propagate error from layer below
        with torch.no_grad():
            if e_below is not None:
                if e_below.dim() == 4:
                    e_below = e_below.flatten(1)
                e_below = e_below * self.in_sign_correct
                update = self.propagate(e_below * d_pred)
                state['x'] += self.gamma * update

            e = state['e'][:, :self.out_features] - state['e'][:, self.out_features:]
            state['x'] += self.gamma * -e

            if temp is not None:
                eps = torch.randn_like(state['x'], device=self.device) * temp * 0.034
                state['x'] += eps