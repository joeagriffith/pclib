from pclib.nn.layers import Conv2dBc
from pclib.nn.models import ConvClassifierUs
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

# Based on Whittington and Bogacz 2017
class ConvClassifierUsBc(ConvClassifierUs):
    """
    | Similar to the FCClassifier, except uses convolutions instead of fully connected layers.
    | Currently calculates X updates using .backward() on the VFE, which is slow.
    | May be possible to speed up by calculating X updates manually, but requires complex indexing to minimise floating point operations.
    | This network is not currently customisable, but requires altering the init_layers() code to change the architecture.

    Args:
        | steps (int): Number of steps to run the network for.
        | bias (bool): Whether to include bias terms in the network.
        | symmetric (bool): Whether to use symmetric weights. 
        | precision_weighted (bool): Whether to weight the VFE by the precision of the prediction.
        | actv_fn (function): Activation function to use in the network.
        | d_actv_fn (function): Derivative of the activation function to use in the network.
        | gamma (float): step size for x updates
        | device (torch.device): Device to run the network on.
        | dtype (torch.dtype): Data type to use for network parameters.
    
    Attributes:
        | num_classes (int): Number of classes in the dataset.
        | steps (int): Number of steps to run the network for.
        | device (torch.device): Device to run the network on.
        | factory_kwargs (dict): Keyword arguments for the layers.
        | layers (torch.nn.ModuleList): List of layers in the network.

    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        super().__init__(
            steps=steps,
            bias=bias,
            symmetric=symmetric,
            precision_weighted=precision_weighted,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            device=device,
            dtype=dtype
        )

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(Conv2dBc(None, (1, 32, 32),                  **self.factory_kwargs))
        layers.append(Conv2dBc((1, 32, 32), (32, 16, 16), 5, 2, 2, **self.factory_kwargs))
        layers.append(Conv2dBc((32, 16, 16), (64, 8, 8),  3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2dBc((64, 8, 8), (64, 4, 4),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2dBc((64, 4, 4), (64, 2, 2),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2dBc((64, 2, 2), (64, 1, 1),    3, 2, 1, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.layers[-1].shape[0], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

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
            e_below = state[i-1]['e'] if i > 0 else None
            pred = self.layers[i+1].predict(state[i+1]) if i < len(self.layers) - 1 else None
            if i > 0 or obs is None: # Don't update bottom x if obs is given
                if i < len(self.layers) - 1 or y is None: # Don't update top x if y is given
                    with torch.no_grad():
                        layer.update_x(state[i], e_below, pred, temp=temp)
            if i < len(self.layers) - 1:
                layer.update_e(state[i], pred, temp=temp)