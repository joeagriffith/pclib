import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from typing import Optional
import math

from pclib.nn.layers import FC




class FCSym(FC):
    """
    | This layer renames 'e' as 'e_u' and introduces a second population of error neurons below 'x' called 'e_l'.
    | This layer has separate weights to generate predictions for both the layer above and below.
    | FCSym can also receive predictions from both the layer above and below, which it uses to calculate e_u and e_l.
    | Therefore, this layer is functionally symmetrical, in terms of how it handles predictions and errors bottom-up and top-down.
    | However, it is still optional (symmetric:bool) whether to use the same weights for prediction generation and error propagation.

    Args:
        | in_features (int): Number of input features.
        | out_features (int): Number of output features.
        | has_bias (bool): Whether to include a bias term.
        | symmetric (bool): Whether to reuse prediction weights, for error propagation.
        | actv_fn (callable): Activation function to use.
        | d_actv_fn (callable): Derivative of activation function to use (if None, will be inferred from actv_fn).
        | gamma (float): step size for x updates.
        | device (torch.device): Device to use for computation.
        | dtype (torch.dtype): Data type to use for computation.

    Attributes:
        | V (torch.Tensor): Weights for predicting 'x' in layer below (if not input layer)
        | V_b (torch.Tensor): Bias term for predicting 'x' in layer below (if has_bias=True)
        | V_bu (torch.Tensor): Weights for propagating up error from layer below (if symmetric=False)

        | W (torch.Tensor): Weights for predicting 'x' in layer above (if not output layer)
        | W_b (torch.Tensor): Bias term for predicting 'x' in layer above (if has_bias=True)
        | W_bu (torch.Tensor): Weights for propagating down error from layer above (if symmetric=False)


    """
    __constants__ = ['in_features', 'out_features', 'next_features']
    next_features: Optional[int]
    weight_td: None
    weight_bu: None
    bias: None
    V: Optional[Tensor] # Predicts downwards
    V_b: Optional[Tensor]
    V_bu: Optional[Tensor]
    W: Optional[Tensor] # Predicts upwards
    W_b: Optional[Tensor]
    W_bu: Optional[Tensor]

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 next_features: int,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = F.relu,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.next_features = next_features
        super().__init__(in_features, out_features, has_bias, symmetric, actv_fn, d_actv_fn, gamma, **factory_kwargs)


    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on init args.
        | Only builds V tensors if not input layer, and W tensors if not output layer.
        """

        if self.in_features is not None:
            self.V = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
            self.V.data *= 0.1
            if self.has_bias:
                self.V_b = Parameter(torch.empty(self.in_features, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.V.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.V_b, -bound, bound)
            else:
                self.register_parameter('bias', None)
            if not self.symmetric:
                self.V_bu = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
                nn.init.kaiming_uniform_(self.V_bu, a=math.sqrt(5))
                self.V_bu.data *= 0.1
            else:
                self.register_parameter('V_bu', None)
        else:
            self.register_parameter('V', None)
            self.register_parameter('V_bu', None)
            self.register_parameter('V_b', None)
        if self.next_features is not None:
            self.W = Parameter(torch.empty((self.next_features, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            self.W.data *= 0.1
            if self.has_bias:
                self.W_b = Parameter(torch.empty(self.next_features, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.W_b, -bound, bound)
            else:
                self.register_parameter('W_b', None)
            if not self.symmetric:
                self.W_bu = Parameter(torch.empty((self.out_features, self.next_features), **self.factory_kwargs))
                nn.init.kaiming_uniform_(self.W_bu, a=math.sqrt(5))
                self.W_bu.data *= 0.1
            else:
                self.register_parameter('W_bu', None)
        else:
            self.register_parameter('W', None)
            self.register_parameter('W_bu', None)
            self.register_parameter('W_b', None)
            
    def init_state(self, batch_size):
        """
        | Builds a new state dictionary for the layer.
        | Different from FC in that it has two error tensors, 'e_u' and 'e_l'.
        | 'e_u' is equivalent to FC's 'e', and 'e_l' is the error between 'x' and its prediction from the layer below.

        Args:
            | batch_size (int): Batch size of the state.

        Returns:
            | state (dict): Dictionary containing 'x', 'e_u', and 'e_l' tensors of shape (batch_size, out_features).
        """
        return {
            'x': torch.zeros((batch_size, self.out_features), device=self.device),
            'e_u': torch.zeros((batch_size, self.out_features), device=self.device),
            'e_l': torch.zeros((batch_size, self.out_features), device=self.device),
        }

    # Override predict and propagate to use V and W
    def predict(self, state):
        """
        | Overrides FC.predict() to prevent its use.
        | Use predict_up or predict_down instead.
        """
        raise NotImplementedError("Use predict_up or predict_down instead")
    def propagate(self, e_below):
        """
        | Overrides FC.propagate() to prevent its use.
        | Use propagate_up or propagate_down instead.
        """
        raise NotImplementedError("Use propagate_up or propagate_down instead")

    # Sends preds to layer below, and propagates up the resultant error
    def predict_down(self, state):
        """
        | Uses V and V_b to calculate a prediction of state['x'] in the layer below.

        Args:
            | state (dict): Dictionary containing 'x', 'e_u', and 'e_l' tensors for this layer.

        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer below.
        """
        return F.linear(self.actv_fn(state['x'].detach()), self.V, self.V_b)
    def propagate_up(self, e_below):
        """
        | Propagates error from layer below, returning an update signal for state['x'].

        Args:
            | e_below (torch.Tensor): Error from layer below

        Returns:
            | update (torch.Tensor): Update signal for state['x'].
        """
        V_bu = self.V.T if self.symmetric else self.V_bu
        return F.linear(e_below, V_bu, None)

    # Sends preds to layer above, and propagates down the resultant error 
    def predict_up(self, state):
        """
        | Uses W and W_b to calculate a prediction of state['x'] in the layer above.

        Args:
            | state (dict): Dictionary containing 'x', 'e_u', and 'e_l' tensors for this layer.

        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer above.
        """
        return F.linear(self.actv_fn(state['x'].detach()), self.W, self.W_b)
    def propagate_down(self, e_above):
        """
        | Propagates error from layer above, returning an update signal for state['x'].

        Args:
            | e_above (torch.Tensor): Error from layer above.

        Returns:
            | update (torch.Tensor): Update signal for state['x'].
        """
        W_bu = self.W.T if self.symmetric else self.W_bu
        return F.linear(e_above, W_bu, None)
        
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, bu_pred=None, td_pred=None, temp=None):
        """
        | Updates state['e_u'] and state['e_l'] inplace between state['x'] and the top-down and bottom-up predictions of it.
        | Uses simulated annealing if temp is not None.

        Args:
            | state (dict): Dictionary containing 'x', 'e_u', and 'e_l' tensors for this layer.
            | bu_pred (Optional[torch.Tensor]): Bottom-up prediction of state['x'].
            | td_pred (Optional[torch.Tensor]): Top-down prediction of state['x'].
            | temp (Optional[float]): Temperature for simulated annealing.
        """
        if bu_pred is not None:
            if bu_pred.dim() == 4:
                bu_pred = bu_pred.flatten(1)
            state['e_l'] = state['x'].detach() - bu_pred
        if td_pred is not None:
            if td_pred.dim() == 4:
                td_pred = td_pred.flatten(1)
            state['e_u'] = state['x'].detach() - td_pred

        if temp is not None:
            eps_l = torch.randn_like(state['e_l'].detach(), device=self.device) * 0.034 * temp
            eps_u = torch.randn_like(state['e_u'].detach(), device=self.device) * 0.034 * temp
            state['e_l'] += eps_l
            state['e_u'] += eps_u
    
    def update_x(self, state, e_below=None, e_above=None):
        """
        | Updates state['x'] inplace, using the error signals from the layer above and below.

        Args:
            | state (dict): Dictionary containing 'x', 'e_u', and 'e_l' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. None if input layer.
            | e_above (Optional[torch.Tensor]): Error of layer above. None if output layer.
        """
        if e_below is not None:
            update = self.propagate_up(e_below)
            state['x'] += self.gamma * (-state['e_l'] + update * self.d_actv_fn(state['x']))
        if e_above is not None:
            update = self.propagate_down(e_above)
            state['x'] += self.gamma * (-state['e_u'] + update * self.d_actv_fn(state['x']))
        
    def assert_grad(self, state, e_below=None, e_above=None):
        """
        | Not implemented.
        """
        raise(NotImplementedError)