import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional
from pclib.utils.functional import my_relu, reTanh

def create_competition_matrix(z_dim, n_group, beta_scale=1.0, alpha_scale=1.0):
    """
    COPIED FROM: code for 'The Predictive Forward-Forward Algorithm' by Ororbia, Mali 2023
    https://github.com/ago109/predictive-forward-forward/blob/adeb918941afaafb11bc9f1b0953dae2d7dd1f13/src/pff_rnn.py#L151
    """
    diag = torch.eye(z_dim)
    V_l = None
    g_shift = 0
    while (z_dim - (n_group + g_shift)) >= 0:
        if g_shift > 0:
            left = torch.zeros([1,g_shift])
            middle = torch.ones([1,n_group])
            right = torch.zeros([1,z_dim - (n_group + g_shift)])
            slice = torch.concat([left,middle,right],axis=1)
            for n in range(n_group):
                V_l = torch.concat([V_l,slice],axis=0)
        else:
            middle = torch.ones([1,n_group])
            right = torch.zeros([1,z_dim - n_group])
            slice = torch.concat([middle,right],axis=1)
            for n in range(n_group):
                if V_l is not None:
                    V_l = torch.concat([V_l,slice],axis=0)
                else:
                    V_l = slice
        g_shift += n_group
    V_l = V_l * (1.0 - diag) * beta_scale + diag * alpha_scale
    return V_l

class FCLI(nn.Module):
    __constants__ = ['size', 'prev_size']
    shape: int
    prev_shape: Optional[int]
    weight_td: Optional[Tensor]
    weight_bu: Optional[Tensor]
    bias: Optional[Tensor]

    def __init__(self,
                 shape: int,
                 prev_shape: int = None,
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
        super(FCLI, self).__init__()

        self.shape = shape
        self.prev_shape = prev_shape
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.beta = beta
        self.device = device
        self.lat_conn_mat = (create_competition_matrix(shape, 10) * (2*torch.eye(shape) - 1)).to(device)


        # Default derivative of activation function
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
        
        # Initialise weights
        # self.weight_lat = Parameter(torch.eye((shape), **factory_kwargs) * 0.3)
        self.weight_lat = Parameter(torch.ones((shape,shape), **factory_kwargs) * 0.15)
        if prev_shape is not None:
            self.weight_td = Parameter(torch.empty((prev_shape, shape), **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(prev_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            if not symmetric:
                self.weight_bu = Parameter(torch.empty((shape, prev_shape), **factory_kwargs))
            else:
                self.register_parameter('weight_bu', None)
        else:
            self.register_parameter('weight_td', None)
            self.register_parameter('weight_bu', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight_td is not None:
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
            'x': torch.zeros((batch_size, self.shape), device=self.device),
            'e': torch.zeros((batch_size, self.shape), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.lat_conn_mat = self.lat_conn_mat.to(self.device)
        return super().to(*args, **kwargs)

    def _lateral(self, state):
        actv = self.actv_fn(state['x'])
        lat_connectivity = self.lat_conn_mat * F.relu(self.weight_lat) # self-excitation, lateral-inhibition, and no negative weights
        return F.linear(actv, lat_connectivity, None)

    # Returns a prediction of the state in the previous layer
    def predict(self, state):
        return F.linear(self.actv_fn(state['x'].detach()), self.weight_td, self.bias)
    
    # propogates error from layer below, return an update for x
    def propagate(self, e_below):
        weight_bu = self.weight_td.T if self.symmetric else self.weight_bu
        return F.linear(e_below, weight_bu, None)
        
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, pred=None, temp=None):
        if pred is not None:
            if pred.dim() == 4:
                pred = pred.flatten(1)
            state['e'] = state['x'].detach() - pred

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * 0.034 * temp
            state['e'] += eps

    def update_x(self, state, e_below=None):
        # If not input layer, propagate error from layer below
        lateral = self._lateral(state)
        if e_below is not None:
            update = self.propagate(e_below)
            state['x'] += self.gamma * (lateral - state['e'] + update * (self.d_actv_fn(state['x'])))
        else:
            state['x'] += self.gamma * (lateral - state['e'])
        
    def assert_grad(self, state, e_below=None):
        with torch.no_grad():
            assert (e_below is None) == (self.prev_shape is None), "e_below must be None iff prev_shape is None"
            if e_below is not None:
                b_size = e_below.shape[0]
                true_weight_td_grad = -(e_below.T @ self.actv_fn(state['x'])) / b_size
                assert torch.eq(F.normalize(self.weight_td.grad, dim=(0,1)), F.normalize(true_weight_td_grad, dim=(0,1))).all(), f"true: {true_weight_td_grad}, backprop: {self.weight_td.grad}"
                assert torch.eq(self.weight_td.grad.norm(dim=(0,1)), true_weight_td_grad.norm(dim=(0,1))).all(), f"true: {true_weight_td_grad.norm(dim=(0,1))}, backprop: {self.weight_td.grad.norm(dim=(0,1))}"
                assert torch.eq(self.weight_td.grad, true_weight_td_grad).all(), f"true: {true_weight_td_grad}, backprop: {self.weight_td.grad}"
                if self.bias is not None:
                    true_bias_grad = -e_below.mean(dim=0)
                    assert torch.eq(F.normalize(self.bias.grad, dim=0), F.normalize(true_bias_grad, dim=0)).all(), f"true: {true_bias_grad}, backprop: {self.bias.grad}"
                    assert torch.eq(self.bias.grad.norm(), true_bias_grad.norm()).all(), f"true: {true_bias_grad.norm()}, backprop: {self.bias.grad.norm()}"
                    assert torch.eq(self.bias.grad, true_bias_grad).all(), f"true: {true_bias_grad}, backprop: {self.bias.grad}"
                if not self.symmetric:
                    true_weight_bu_grad = -(self.actv_fn(state['x']).T @ e_below) / b_size
                    assert torch.eq(F.normalize(self.weight_bu.grad, dim=(0,1)), F.normalize(true_weight_bu_grad, dim=(0,1))).all(), f"true: {true_weight_bu_grad}, backprop: {self.weight_bu.grad}"
                    assert torch.eq(self.weight_bu.grad.norm(dim=(0,1)), true_weight_bu_grad.norm(dim=(0,1))).all(), f"true: {true_weight_bu_grad.norm(dim=(0,1))}, backprop: {self.weight_bu.grad.norm(dim=(0,1))}"
                    assert torch.eq(self.weight_bu.grad, true_weight_bu_grad).all()
        return True