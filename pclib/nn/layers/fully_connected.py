import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional
from pclib.utils.functional import reTanh

class FC(nn.Module):
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

        self.init_params()
        
    # Declare weights if not input layer
    def init_params(self):
        """
        | Creates and initialises weight tensors and bias tensor based on init args.
        """
        if self.in_features is not None:
            self.weight_td = Parameter(torch.empty((self.in_features, self.out_features), **self.factory_kwargs))
            nn.init.kaiming_uniform_(self.weight_td, a=math.sqrt(5))
            self.weight_td.data *= 0.1

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
                self.weight_bu.data *= 0.1
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
            state['e'] = state['x'].detach() - pred

        if temp is not None:
            eps = torch.randn_like(state['e'].detach(), device=self.device) * 0.034 * temp
            state['e'] += eps
    
    def update_x(self, state, e_below=None):
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
            state['x'] += self.gamma * (-state['e'] + update * self.d_actv_fn(state['x']))
        else:
            state['x'] += self.gamma * (-state['e'])

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


# TODO: Confirm descriptions for beta_scale and alpha_scale are correct.
def create_competition_matrix(z_dim, n_group, beta_scale=1.0, alpha_scale=1.0):
    """
    | COPIED FROM: code for 'The Predictive Forward-Forward Algorithm' by Ororbia, Mali 2023
    | https://github.com/ago109/predictive-forward-forward/blob/adeb918941afaafb11bc9f1b0953dae2d7dd1f13/src/pff_rnn.py#L151
    | Builds a symmetric binary matrix which splits the z_dim neurons into n_group even groups.
    | Members of a group are connected only to each other, with a weight of 1.0.

    Args:
        | z_dim (int): Number of neurons in layer.
        | n_group (int): Number of groups to split neurons into. Must be a factor of z_dim.
        | beta_scale (float): Scaling factor for connections between neurons
        | alpha_scale (float): Scaling factor for self-connections of neurons
    
    Returns:
        | V_l (torch.Tensor): Symmetric binary matrix of shape (z_dim, z_dim).

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

class FCLI(FC):
    """
    | Fully connected layer with optional bias and optionally symmetric weights.
    | This layer inherits most of its functionality from FC. 
    | However, It overrides the update_x(), predict() and propagate() to use lateral connections.
    | Implements a K-Winner-Take-All lateral inhibition scheme, with self-excitation.
    | Suggest using reTanh as the activation function.
    | The use of lateral inhibition is hoped to enforce sparsity in the layer's representations.

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
        | lat_conn_mat (torch.Tensor): Lateral connectivity binary matrix. Not an optimisable parameter.
        | weight_lat (torch.Tensor): Lateral weights for self-excitation and lateral inhibition.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int = None,
                 has_bias: bool = True,
                 symmetric: bool = True,
                 actv_fn: callable = reTanh,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, has_bias, symmetric, actv_fn, d_actv_fn, gamma, **factory_kwargs)
        self.lat_conn_mat = (create_competition_matrix(out_features, 10) * (2*torch.eye(out_features) - 1)).to(device)
        
        # Initialise weights
        self.weight_lat = Parameter(torch.eye((out_features), **factory_kwargs) * 1.0)

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.lat_conn_mat = self.lat_conn_mat.to(self.device)
        return super().to(*args, **kwargs)

    # Calculates a prediction of state['x'] in the layer below
    def predict(self, state):
        """
        | Almost identical to FC.predict(), but doesn't apply the activation function to state['x'] before calculating the prediction.
        | This is because the activation function is applied to state['x'] in the update_x() function.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.

        Returns:
            | pred (torch.Tensor): Prediction of state['x'] in the layer below.
        """
        return F.linear(state['x'].detach(), self.weight_td, self.bias)

    def lateral(self, state):
        """
        | Calculates the lateral update signal for state['x'].
        | The new state['x'] is calculated as the interpolation between the current state['x'] and lateral(state['x'])

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.

        Returns:
            | new_x (torch.Tensor): 
        """
        lat_connectivity = self.lat_conn_mat * F.relu(self.weight_lat) # self-excitation, lateral-inhibition, and no negative weights
        return F.linear(self.actv_fn(state['x']), lat_connectivity, None)
        
    def update_x(self, state, e_below=None):
        """
        | Calculates a new_x and then interpolates between the current state['x'] and new_x, updating state['x'] inplace.
        | This uses the lateral connectivity to produce a target value, rather than an incremental update.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below Optional([torch.Tensor]): Error of layer below. if None, no gradients are calculated.
        """
        new_x = self.lateral(state)
        if e_below is not None:
            update = self.propagate(e_below)
            new_x += update * (self.d_actv_fn(state['x'])) - state['e']
        else:
            new_x += -state['e']
        state['x'] = (1-self.gamma) * state['x'] + self.gamma * self.actv_fn(new_x)
        
    def assert_grad(self, state, e_below=None):
        raise(NotImplementedError)

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