import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCSym
from pclib.nn.models import FCClassifierInv

class FCClassifierInvSym(FCClassifierInv):
    """
    | Inherits most functionality from FCClassifierInv, except it uses FCSym layers instead of FC layers.
    | In theory is functionally symmetrical. Predictions flow both ways, and same for error propagations.
    | Currently does not learn, but may just need more hyperparameter tuning.
    
    Args:
        | in_features (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
        | actv_fn (torch.nn.functional): Activation function to use
        | d_actv_fn (torch.nn.functional): Derivative of activation function to use
        | gamma (float): step size for x updates
        | device (torch.device): Device to run on
        | dtype (torch.dtype): Data type to use

    Attributes:
        | layers (torch.nn.ModuleList): List of layers
        | in_features (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
    """
    def __init__(self, in_features, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Identical to FCClassifierInv, except it uses FCSym layers instead of FC layers.
        """
        layers = []
        features = [self.num_classes] + self.hidden_sizes + [self.in_features]
        for i, out_features in enumerate(features):
            in_features = features[i-1] if i > 0 else None
            next_features = features[i+1] if i < len(features) - 1 else None
            if self.precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCSym layers")
            else:
                layers.append(FCSym(in_features, out_features, next_features, device=self.device, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

    def vfe(self, state, batch_reduction='mean', layer_reduction='sum'):
        """
        | Calculates the Variational Free Energy (VFE) of the model.
        | Overrides FCClassifierInv.vfe() as it has to encorporate both 'e_l' and 'e_u' in vfe calculation.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'
            | batch_reduction (str): How to reduce over batches ['sum', 'mean']
            | layer_reduction (str): How to reduce over layers ['sum', 'mean']
        
        Returns:
            | vfe (torch.Tensor): VFE of the model (scalar)
        """
        if layer_reduction == 'sum':
            vfe = sum([state_i['e_l'].square().sum(dim=[i for i in range(1, state_i['e_l'].dim())]) for state_i in state]) + sum([state_i['e_u'].square().sum(dim=[i for i in range(1, state_i['e_u'].dim())]) for state_i in state])
        elif layer_reduction =='mean':
            vfe = sum([state_i['e_l'].square().mean(dim=[i for i in range(1, state_i['e_l'].dim())]) for state_i in state]) + sum([state_i['e_u'].square().mean(dim=[i for i in range(1, state_i['e_u'].dim())]) for state_i in state])
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()
        return vfe
        
    def step(self, state, obs=None, y=None, temp=None):
        """
        | Performs one step of inference. Updates Es first, then Xs, then pins.
        | Es are update bottom-up, using predictions from both layer above and layer below, if they exist.
        | Xs are also updated bottom-up, using Es from layer below and layer above, if they exist.
        | Pins are performed last on the input and/or target if provided.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | temp (Optional[float]): Temperature to use for update
        """

        # Update Es, Top-down so we can collect predictions as we descend
        for i, layer in enumerate(self.layers):
            bu_pred = self.layers[i-1].predict_up(state[i-1]) if i > 0 else None
            td_pred = self.layers[i+1].predict_down(state[i+1]) if i < len(self.layers) - 1 else None
            layer.update_e(state[i], bu_pred, td_pred, temp=temp)

        # Update Xs
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                e_below = state[i-1]['e_u'] if i > 0 else None
                e_above = state[i+1]['e_l'] if i < len(self.layers) - 1 else None
                layer.update_x(state[i], e_below, e_above)
        
        self.pin(state, obs, y)
        
    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs dependant using either obs or y if provided.
        | Similar to FCClassifierInv._init_xs(), but uses predict_down() and predict_up() instead of predict() and propagate().

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict_down(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            pred = y.clone()
            for i, layer in enumerate(self.layers):
                state[i]['x'] = pred
                pred = layer.predict_up(state[i]) if i < len(self.layers) - 1 else None
                # if i == 0:
                #     state[0]['x'] = y.clone()
                # else:
                #     state[i]['x'] = layer.predict_up(state[i-1])

    def assert_grads(model, state):
        """
        | Uses assertions to compare current gradients of each layer to manually calculated gradients.
        | Only useful if using autograd=True in training, otherwise comparing manual grads to themselves.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'
        """
        for i, layer in enumerate(model.layers):
            e_m_1 = state[i-1]['e'] if i > 0 else None
            e_p_1 = state[i+1]['e'] if i < len(model.layers) - 1 else None
            layer.assert_grad(state[i], e_m_1, e_p_1)