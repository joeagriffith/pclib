import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

# Whittington & Bogacz 2017
class LinearSym(nn.Module):
    __constants__ = ['shape', 'prev_shape', 'next_shape']
    shape: int
    prev_shape: Optional[int]
    next_shape: Optional[int]
    V: Optional[Tensor] # Predicts downwards
    V_b: Optional[Tensor]
    V_bu: Optional[Tensor]
    W: Optional[Tensor] # Predicts upwards
    W_b: Optional[Tensor]
    W_bu: Optional[Tensor]

    def __init__(self,
                 shape: int,
                 prev_shape: int = None,
                 next_shape: int = None,
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
        super(LinearSym, self).__init__()

        self.shape = shape
        self.prev_shape = prev_shape
        self.next_shape = next_shape
        self.symmetric = symmetric
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.beta = beta
        self.device = device


        # Default derivative of activation function
        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()
        
        # Initialise weights if not input layer
        if prev_shape is not None:
            self.V = Parameter(torch.empty((prev_shape, shape), **factory_kwargs))
            if bias:
                self.V_b = Parameter(torch.empty(prev_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            if not symmetric:
                self.V_bu = Parameter(torch.empty((shape, prev_shape), **factory_kwargs))
            else:
                self.register_parameter('V_bu', None)
        else:
            self.register_parameter('V', None)
            self.register_parameter('V_bu', None)
            self.register_parameter('V_b', None)
        if next_shape is not None:
            self.W = Parameter(torch.empty((next_shape, shape), **factory_kwargs))
            if bias:
                self.W_b = Parameter(torch.empty(next_shape, **factory_kwargs))
            else:
                self.register_parameter('W_b', None)
            if not symmetric:
                self.W_bu = Parameter(torch.empty((shape, next_shape), **factory_kwargs))
            else:
                self.register_parameter('W_bu', None)
        else:
            self.register_parameter('W', None)
            self.register_parameter('W_bu', None)
            self.register_parameter('W_b', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.prev_shape is not None:
            nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
            self.V.data *= 0.1
            if self.V_b is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.V.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.V_b, -bound, bound)
            if not self.symmetric:
                nn.init.kaiming_uniform_(self.V_bu, a=math.sqrt(5))
                self.V_bu.data *= 0.1
        if self.next_shape is not None:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
            self.W.data *= 0.1
            if self.W_b is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W.T)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.W_b, -bound, bound)
            if not self.symmetric:
                nn.init.kaiming_uniform_(self.W_bu, a=math.sqrt(5))
                self.W_bu.data *= 0.1
            
    def init_state(self, batch_size):
        return {
            'x': torch.zeros((batch_size, self.shape), device=self.device),
            'e_u': torch.zeros((batch_size, self.shape), device=self.device),
            'e_l': torch.zeros((batch_size, self.shape), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    # Returns a prediction of the state in the previous layer
    def predict_down(self, state):
        return F.linear(self.actv_fn(state['x'].detach()), self.V, self.V_b)
    
    # propogates error from layer below, return an update for x
    def propagate_up(self, e_below):
        V_bu = self.V.T if self.symmetric else self.V_bu
        return F.linear(e_below, V_bu, None)

    # Returns a prediction of the state in the next layer
    def predict_up(self, state):
        return F.linear(self.actv_fn(state['x'].detach()), self.W, self.W_b)
    
    # propogates error from layer above, return an update for x
    def propagate_down(self, e_below):
        W_bu = self.W.T if self.symmetric else self.W_bu
        return F.linear(e_below, W_bu, None)
        
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, bu_pred=None, td_pred=None, temp=None):
        if bu_pred is not None:
            if bu_pred.dim() == 4:
                if bu_pred.shape[2] == 1 and bu_pred.shape[3] == 1:
                    bu_pred = bu_pred.squeeze(-1).squeeze(-1)
                else:
                    raise ValueError("Prediction must be 2D, or 4D with 1x1 spatial dimensions")
            state['e_l'] = state['x'].detach() - bu_pred
        if td_pred is not None:
            if td_pred.dim() == 4:
                if td_pred.shape[2] == 1 and td_pred.shape[3] == 1:
                    td_pred = td_pred.squeeze(-1).squeeze(-1)
                else:
                    raise ValueError("Prediction must be 2D, or 4D with 1x1 spatial dimensions")
            state['e_u'] = state['x'].detach() - td_pred

        if temp is not None:
            eps_l = torch.randn_like(state['e_l'].detach(), device=self.device) * 0.034 * temp
            eps_u = torch.randn_like(state['e_u'].detach(), device=self.device) * 0.034 * temp
            state['e_l'] += eps_l
            state['e_u'] += eps_u
    
    def update_x(self, state, e_below=None, e_above=None):
        # If not input layer, propagate error from layer below
        if e_below is not None:
            update = self.propagate_up(e_below)
            state['x'] += self.gamma * (0.25 * -state['e_l'] + update * self.d_actv_fn(state['x']))
        if e_above is not None:
            update = self.propagate_down(e_above)
            state['x'] += self.gamma * (0.25 * -state['e_u'] + update * self.d_actv_fn(state['x']))
        
    def assert_grad(self, state, e_below=None, e_above=None):
        with torch.no_grad():
            assert (e_below is None) == (self.prev_shape is None), "e_below must be None iff prev_shape is None"
            assert (e_above is None) == (self.next_shape is None), "e_above must be None iff next_shape is None"
            if e_below is not None:
                b_size = e_below.shape[0]
                true_V_grad = -(e_below.T @ self.actv_fn(state['x'])) / b_size
                assert torch.eq(F.normalize(self.V.grad, dim=(0,1)), F.normalize(true_V_grad, dim=(0,1))).all(), f"true: {true_V_grad}, backprop: {self.V.grad}"
                assert torch.eq(self.V.grad.norm(dim=(0,1)), true_V_grad.norm(dim=(0,1))).all(), f"true: {true_V_grad.norm(dim=(0,1))}, backprop: {self.V.grad.norm(dim=(0,1))}"
                assert torch.eq(self.V.grad, true_V_grad).all(), f"true: {true_V_grad}, backprop: {self.V.grad}"
                if self.V_b is not None:
                    true_V_b_grad = -e_below.mean(dim=0)
                    assert torch.eq(F.normalize(self.V_b.grad, dim=0), F.normalize(true_V_b_grad, dim=0)).all(), f"true: {true_V_b_grad}, backprop: {self.V_b.grad}"
                    assert torch.eq(self.V_b.grad.norm(), true_V_b_grad.norm()).all(), f"true: {true_V_b_grad.norm()}, backprop: {self.V_b.grad.norm()}"
                    assert torch.eq(self.V_b.grad, true_V_b_grad).all(), f"true: {true_V_b_grad}, backprop: {self.V_b.grad}"
                if not self.symmetric:
                    true_V_bu_grad = -(self.actv_fn(state['x']).T @ e_below) / b_size
                    assert torch.eq(F.normalize(self.V_bu.grad, dim=(0,1)), F.normalize(true_V_bu_grad, dim=(0,1))).all(), f"true: {true_V_bu_grad}, backprop: {self.V_bu.grad}"
                    assert torch.eq(self.V_bu.grad.norm(dim=(0,1)), true_V_bu_grad.norm(dim=(0,1))).all(), f"true: {true_V_bu_grad.norm(dim=(0,1))}, backprop: {self.V_bu.grad.norm(dim=(0,1))}"
                    assert torch.eq(self.V_bu.grad, true_V_bu_grad).all()
            if e_above is not None:
                b_size = e_above.shape[0]
                true_W_grad = -(e_above.T @ self.actv_fn(state['x'])) / b_size
                assert torch.eq(F.normalize(self.W.grad, dim=(0,1)), F.normalize(true_W_grad, dim=(0,1))).all(), f"true: {true_W_grad}, backprop: {self.W.grad}"
                assert torch.eq(self.W.grad.norm(dim=(0,1)), true_W_grad.norm(dim=(0,1))).all(), f"true: {true_W_grad.norm(dim=(0,1))}, backprop: {self.W.grad.norm(dim=(0,1))}"
                assert torch.eq(self.W.grad, true_W_grad).all(), f"true: {true_W_grad}, backprop: {self.W.grad}"
                if self.W_b is not None:
                    true_W_b_grad = -e_above.mean(dim=0)
                    assert torch.eq(F.normalize(self.W_b.grad, dim=0), F.normalize(true_W_b_grad, dim=0)).all(), f"true: {true_W_b_grad}, backprop: {self.W_b.grad}"
                    assert torch.eq(self.W_b.grad.norm(), true_W_b_grad.norm()).all(), f"true: {true_W_b_grad.norm()}, backprop: {self.W_b.grad.norm()}"
                    assert torch.eq(self.W_b.grad, true_W_b_grad).all(), f"true: {true_W_b_grad}, backprop: {self.W_b.grad}"
                if not self.symmetric:
                    true_W_bu_grad = -(self.actv_fn(state['x']).T @ e_above) / b_size
                    assert torch.eq(F.normalize(self.W_bu.grad, dim=(0,1)), F.normalize(true_W_bu_grad, dim=(0,1))).all(), f"true: {true_W_bu_grad}, backprop: {self.W_bu.grad}"
                    assert torch.eq(self.W_bu.grad.norm(dim=(0,1)), true_W_bu_grad.norm(dim=(0,1))).all(), f"true: {true_W_bu_grad.norm(dim=(0,1))}, backprop: {self.W_bu.grad.norm(dim=(0,1))}"
                    assert torch.eq(self.W_bu.grad, true_W_bu_grad).all(), f"true: {true_W_bu_grad}, backprop: {self.W_bu.grad}"
        return True