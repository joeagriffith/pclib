import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import FCDP
from pclib.nn.models import FCClassifierUs

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierUsDp(FCClassifierUs):
    """
    | A Self-Supervised version of FCClassifier.
    | It learns a feature extractor (self.layers) only from observations.
    | It separately learns a classifier (self.classifier) which takes the output of self.layers as input.

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
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC) (NOT IMPLEMENTED)
        | classifier (torch.nn.Sequential): Classifier to use
    """
    def __init__(self, in_features, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(in_features, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        for out_features in [self.in_features] + self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError
                # layers.append(FCPW(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(FCDP(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs using either y or obs if provided.
        | If y is provided, then top down predictions are calculated and used as initial xs.
        | Else if obs is provided, then bottom up error propagations (pred=0) are calculated and used as initial xs.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if y is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = y.clone()
                else:
                    state[i-1]['x'] = layer.predict(state[i])
                    state[i-1]['x'] = self.layers[i-1].actv_fn(state[i-1]['x'])
            if obs is not None:
                state[0]['x'] = obs.clone()
        elif obs is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = obs.clone()
                else:
                    # state[i]['x'] = layer.actv_fn(layer.propagate(state[i-1]['x']))
                    state[i]['x'] = layer.actv_fn(torch.randn_like(state[i]['x']))



    def vfe(self, state, batch_reduction='mean', unit_reduction='sum'):
        """
        | Calculates the Variational Free Energy (VFE) of the model.
        | This is the sum of the squared prediction errors of each layer.
        | how batches and units are reduced is controlled by batch_reduction and unit_reduction.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | batch_reduction (str): How to reduce over batches ['sum', 'mean', None]
            | unit_reduction (str): How to reduce over units ['sum', 'mean']

        Returns:
            | vfe (torch.Tensor): VFE of the model (scalar)
        """
        # Reduce units for each layer
        Es = [torch.cat([state_i['pPE'], state_i['nPE']], dim=1) for state_i in state]
        if unit_reduction == 'sum':
            vfe = [e.square().sum(dim=1) for e in Es]
        elif unit_reduction =='mean':
            vfe = [e.square().mean(dim=1) for e in Es]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe


    def step(self, state, obs=None, y=None, temp=None):
        """
        | Performs one step of inference. Updates Es first, then Xs, then pins.
        | Es are updated top-down, Xs are updated bottom-up, due to the way the updates flow.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e; (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | temp (Optional[float]): Temperature to use for update

        """
        for i, layer in enumerate(self.layers):
            ppe_below = state[i-1]['pPE'] if i > 0 else None
            npe_below = state[i-1]['nPE'] if i > 0 else None
            if i > 0 or obs is None: # Don't update bottom x if obs is given
                if i < len(self.layers) - 1 or y is None: # Don't update top x if y is given
                    with torch.no_grad():
                        layer.update_x(state[i], ppe_below, npe_below, temp=temp)
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)