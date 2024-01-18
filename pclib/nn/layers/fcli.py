import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FC
from pclib.utils.functional import reTanh

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
        self.group_size = 20

        # self.lat_conn_mat = (create_competition_matrix(out_features, out_features//self.group_size) * (2*torch.eye(out_features) - 1)).to(device)
        self.lat_conn_mat = (create_competition_matrix(out_features, 1) * (2*torch.eye(out_features) - 1)).to(device)
        
        # Initialise lateral weights to zero (this is the offset from identity matrix so we can use standard weight decay)
        self.weight_lat = nn.Parameter(torch.zeros((out_features, out_features), **factory_kwargs))

        # Normalize moving_avg??!?! and on each update
        # self.moving_avg = torch.ones((out_features), **factory_kwargs)
        self.moving_avg = None

    def to(self, *args, **kwargs):
        self.device = args[0]
        self.lat_conn_mat = self.lat_conn_mat.to(self.device)
        return super().to(*args, **kwargs)

    def lateral(self, state):
        """
        | Calculates the lateral update signal for state['x'].
        | The new state['x'] is calculated as the interpolation between the current state['x'] and lateral(state['x'])

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.

        Returns:
            | new_x (torch.Tensor): 
        """
        lat_connectivity = self.lat_conn_mat * F.relu(self.weight_lat + torch.eye(self.out_features, device=self.device)*1.2) # self-excitation, lateral-inhibition, and no negative weights
        # return F.linear(self.actv_fn(self.boost(state['x'])), lat_connectivity, None)
        # return F.linear(self.boost(state['x']), lat_connectivity, None)
        # return F.linear(self.boost(F.relu(state['x'])), lat_connectivity, None)
        return F.linear(F.relu(state['x']), lat_connectivity, None)
    
    def predict(self, state):
        return F.linear(state['x'].detach(), self.weight_td, self.bias)
        
    def update_x(self, state, e_below=None, temp=None):
        """
        | Calculates a new_x and then interpolates between the current state['x'] and new_x, updating state['x'] inplace.
        | This uses the lateral connectivity to produce a target value, rather than an incremental update.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
            | e_below Optional([torch.Tensor]): Error of layer below. if None, no gradients are calculated.
        """
        # state['x'] = (1.0 - self.gamma) * state['x'] + self.gamma * self.lateral(state)
        dx = self.lateral(state)
        if e_below is not None:
            dx += self.propagate(e_below)# * self.d_actv_fn(state['x'].detach())

        dx += 0.34 * -state['e']

        if temp is not None:
            dx += torch.randn_like(state['x'].detach(), device=self.device) * temp * 0.034
        
        # state['x'] = state['x'].detach() + self.gamma * dx
        # state['x'] = (1.0 - self.gamma) * state['x'] + self.gamma * dx
        state['x'] = (1.0 - self.gamma) * state['x'] + self.gamma * self.actv_fn(state['x'] + self.boost(dx))
        
    def assert_grad(self, state, e_below=None):
        raise(NotImplementedError)
    
    def update_mov_avg(self, state):
        """
        | Updates the moving average of the layer's activations.
        | This is used to calculate the variance of the activations, which is used to scale the error signal.

        Args:
            | state (dict): Dictionary containing 'x' and 'e' tensors for this layer.
        """

        if self.moving_avg is None:
            self.moving_avg = state['x'].mean(dim=0)
        else:
            # self.moving_avg = 0.999 * self.moving_avg + 0.001 * state['x'].mean(dim=0)
            self.moving_avg = 0.9 * self.moving_avg + 0.1 * state['x'].mean(dim=0)
    
    def boost(self, x):
        # increases x_i if x_i is lower than the average of the group, and vice versa
        if self.moving_avg is None:
            self.moving_avg = x.mean(dim=0)
        mult = (self.moving_avg * self.lat_conn_mat.abs()).mean(dim=0)  / self.moving_avg
        return x * mult

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