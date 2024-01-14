import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from typing import Optional
from pclib.utils.functional import reTanh, identity
from pclib.nn.layers import FC

class FCPW2(FC):
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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            symmetric=symmetric,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            device=device,
            dtype=dtype,
        )

    def predict(self, state):
        """
        | Calculates a prediction of state['x'] in the layer below.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
        
        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer below.
        """
        x = state['x'].detach() * (1 - state['e'].detach())
        pre_act = F.linear(x, self.weight_td, self.bias)

        return self.actv_fn(pre_act), self.d_actv_fn(pre_act)
    
    
    def propagate(self, e_below):
        """
        | Propagates error from layer below, returning an update signal for state['x'].

        Args:
            | e_below (torch.Tensor): Error signal from layer below.

        Returns:
            | update (torch.Tensor): Update signal for state['x'].
        """
        if e_below.dim() == 4:
            e_below = e_below.flatten(1)
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
            state['e'] = self.actv_fn(state['x'].detach()) - pred

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * 0.034 * temp
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
            dx = torch.zeros_like(state['x'], device=self.device)
            if e_below is not None:
                if e_below.dim() == 4:
                    e_below = e_below.flatten(1)
                update = self.propagate(e_below * d_pred) * (1 - state['e'])
                # saves a tiny bit of compute if d_actv_fn is identity
                dx += update

            dx += 0.1 * 0.34 * -(state['e'] * self.d_actv_fn(state['x'].detach()))

            # dx += 0.1 * 0.1 * -state['x']

            if temp is not None:
                dx += torch.randn_like(state['x'], device=self.device) * temp * 0.034

            state['x'] = state['x'].detach() + self.gamma * dx

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
            if e_below.dim() == 4:
                e_below = e_below.flatten(1)
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