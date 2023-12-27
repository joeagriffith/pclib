import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity

class FCBCDIM(nn.Module):
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
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.device = device

        # Automatically set d_actv_fn if not provided
        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.leaky_relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x)) + torch.sign(torch.minimum(x, torch.zeros_like(x))) * 0.01
        elif actv_fn == reTanh:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x)) * (1 - torch.tanh(x).square())
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()
        elif actv_fn == identity:
            self.d_actv_fn: callable = lambda x: torch.ones_like(x)

        self.init_params()
        self.norm = nn.LayerNorm(self.out_features)
        # self.norm = nn.BatchNorm1d(self.out_features)
        # self.norm = nn.GroupNorm(8, self.out_features)
        
    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on init args.
        """
        if self.in_features is not None:
            self.weight_td = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            # nn.init.kaiming_normal_(self.weight_td, a=math.sqrt(5))
            # nn.init.xavier_uniform_(self.weight_td)
            # nn.init.xavier_normal_(self.weight_td)
            

            if self.has_bias:
                self.bias = Parameter(torch.empty(self.in_features, **self.factory_kwargs))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_td.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter('bias', None)

            if not self.symmetric:
                self.weight_bu = Parameter(torch.empty((self.out_features, self.in_features), **self.factory_kwargs))
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
            'e': torch.zeros((batch_size, self.out_features), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, state):
        """
        | Calculates a prediction of state['x'] in the layer below.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
        
        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer below.
        """

        return F.linear(self.actv_fn(state['x'].detach()), self.weight_td, self.bias)
    
    def propagate(self, e_below):
        """
        | Propagates error from layer below, returning an update signal for state['x'].

        Args:
            | e_below (torch.Tensor): Error signal from layer below.

        Returns:
            | update (torch.Tensor): Update signal for state['x'].
        """
        weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
        return F.linear(e_below, weight_bu, None)
        
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
            state['e'] = state['x'].detach() / (pred + torch.ones_like(pred)*1e-6)

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * 0.034 * temp
            state['e'] += eps
    
    def update_x(self, state, e_below=None, pred=None, temp=None):
        """
        | Updates state['x'] inplace, using the error signal from the layer below and error of the current layer.
        | Formula: new_x = x + gamma * (-e + propagate(e_below) * d_actv_fn(x)).

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. None if input layer.
        """
        # If not input layer, propagate error from layer below
        if e_below is not None:
            update = self.propagate(e_below)
            state['x'] = (state['x'] + torch.ones_like(state['x'])*1e-6) * (update * self.d_actv_fn(state['x']))
        if pred is not None:
            state['x'] = state['x'] * (torch.ones_like(pred) + self.gamma*pred)
        if temp is not None:
            eps = torch.randn_like(state['x'], device=self.device) * 0.034 * temp
            state['x'] += eps
        
        state['x'] = self.norm(state['x'])

    def update_grad(self, state, e_below=None):
        """
        | Manually calculates gradients for weight_td, weight_bu, and bias if they exist.
        | Slightly faster than using autograd.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. if None, no gradients are calculated.
        """
        if e_below is not None:
            b_size = e_below.shape[0]
            self.weight_td.grad = 2*-(e_below.T @ self.actv_fn(state['x'])) / b_size
            if self.bias is not None:
                self.bias.grad = 2*-e_below.mean(dim=0)
            if not self.symmetric:
                self.weight_bu.grad = 2*-(self.actv_fn(state['x']).T @ e_below) / b_size
        
    def assert_grad(self, state, e_below=None):
        """
        | Iff model is being updated with autograd, this function can be used to check whether the manual gradient calculations agree.
        | Uses assertions and torch.isclose to compare.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below (Optional[torch.Tensor]): Error of layer below. if None, no gradients are calculated.
        """
        with torch.no_grad():
            assert (e_below is None) == (self.in_features is None), "e_below must be None iff in_features is None"
            if e_below is not None:
                b_size = e_below.shape[0]
                manual_weight_td_grad = 2*-(e_below.T @ self.actv_fn(state['x'])) / b_size
                isclose = torch.isclose(self.weight_td.grad, manual_weight_td_grad, atol=0.001, rtol=0.1)
                assert isclose.all(), f" \
                    \nbackward: {self.weight_td.grad} \
                    \nmanual  : {manual_weight_td_grad}, \
                    \nrel_diff: {(manual_weight_td_grad - self.weight_td.grad).abs() / manual_weight_td_grad.abs()} \
                    \nrel_diff_max: {((manual_weight_td_grad - self.weight_td.grad).abs() / manual_weight_td_grad.abs()).max()} \
                    \nmax_diff: {(manual_weight_td_grad - self.weight_td.grad).abs().max()} \
                    \n(bak, man, diff): {[(self.weight_td.grad[i, j].item(), manual_weight_td_grad[i, j].item(), (self.weight_td.grad[i, j] - manual_weight_td_grad[i, j]).abs().item()) for i, j in (isclose==False).nonzero()[:5]]}"

                if self.bias is not None:
                    manual_bias_grad = 2*-e_below.mean(dim=0)
                    isclose = torch.isclose(self.bias.grad, manual_bias_grad, atol=0.001, rtol=0.1)
                    assert isclose.all(), f" \
                        \nmanual  : {manual_bias_grad}, \
                        \nbackward: {self.bias.grad} \
                        \nrel_diff: {(manual_bias_grad - self.bias.grad).abs() / manual_bias_grad.abs()} \
                        \nrel_diff_max: {((manual_bias_grad - self.bias.grad).abs() / manual_bias_grad.abs()).max()} \
                        \nmax_diff: {(manual_bias_grad - self.bias.grad).abs().max()} \
                        \n(bak, man, diff): {[(self.bias.grad[i].item(), manual_bias_grad[i].item(), (self.bias.grad[i] - manual_bias_grad[i]).abs().item()) for i in (isclose==False).nonzero()[:5]]}"

                if not self.symmetric:
                    manual_weight_bu_grad = 2*-(self.actv_fn(state['x']).T @ e_below) / b_size
                    isclose = torch.isclose(self.weight_bu.grad, manual_weight_bu_grad, atol=0.001, rtol=0.1)
                    assert isclose.all(), f" \
                        \nmanual  : {manual_weight_bu_grad}, \
                        \nbackward: {self.weight_bu.grad} \
                        \nrel_diff: {(manual_weight_bu_grad - self.weight_bu.grad).abs() / manual_weight_bu_grad.abs()} \
                        \nrel_diff_max: {((manual_weight_bu_grad - self.weight_bu.grad).abs() / manual_weight_bu_grad.abs()).max()} \
                        \nmax_diff: {(manual_weight_bu_grad - self.weight_bu.grad).abs().max()} \
                        \n(bak, man, diff): {[(self.weight_bu.grad[i, j].item(), manual_weight_bu_grad[i, j].item(), (self.weight_bu.grad[i, j] - manual_weight_bu_grad[i, j]).abs().item()) for i, j in (isclose==False).nonzero()[:5]]}"

        return True