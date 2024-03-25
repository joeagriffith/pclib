from pclib.nn.layers import Conv2d, FC
from pclib.nn.models import ConvClassifier
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import List, Optional

# Based on Whittington and Bogacz 2017
class ConvClassifierUs(ConvClassifier):
    """
    | Similar to the ConvClassifer, except it learns an unsupervised feature extractor, and a separate backprop trained classifier.
    | This network is not currently customisable, but requires altering the init_layers() code to change the architecture.

    Parameters
    ----------
        steps : int
            Number of steps to run the network for.
        bias : bool
            Whether to include bias terms in the network.
        symmetric : bool
            Whether to use symmetric weights. 
        actv_fn : callable
            Activation function to use in the network.
        d_actv_fn : Optional[callable]
            Derivative of the activation function to use in the network.
        gamma : float
            step size for x updates
        x_decay : float
            Decay rate for x
        has_memory_vec : bool
            Whether to include a memory vector in the network.
        device : torch.device
            Device to run the network on.
        dtype : torch.dtype
            Data type to use for network parameters.
    """
    def __init__(
            self, 
            in_channels=1,
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.relu, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            x_decay:float = 0.0,
            has_memory_vec:bool = False,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        self.in_channels = in_channels
        self.has_memory_vec = has_memory_vec
        super().__init__(
            steps=steps,
            bias=bias,
            symmetric=symmetric,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            x_decay=x_decay,
            device=device,
            dtype=dtype,
        )

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(Conv2d(None,          (self.in_channels, 32, 32),                  **self.factory_kwargs))
        layers.append(Conv2d((self.in_channels, 32, 32),   (32, 16, 16),  5, 2, 2, **self.factory_kwargs))
        layers.append(Conv2d((32, 16, 16),  (64, 8, 8),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 8, 8),    (128, 4, 4),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((128, 4, 4),    (256, 2, 2),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((256, 2, 2),    (256, 1, 1),    3, 2, 1, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

        if self.has_memory_vec:
            self.memory_vector = nn.Parameter(torch.empty(256, 1, 1, device=self.device, dtype=self.dtype))
            nn.init.normal_(self.memory_vector, mean=0.0, std=0.01)

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        self.memory_vector = nn.Parameter(self.memory_vector.to(device))
        return self


    def reconstruct(self, obs:torch.Tensor = None, steps:int = None):
        """
        | Performs inference for the network, and returns the reconstructed input.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            steps : Optional[int]
                Number of steps to run inference for
        
        Returns
        -------
            torch.Tensor
                Final prediction of input data
            List[dict]
                List of layer state dicts, each containing 'x' and 'e'
        """
        _, state = self.forward(obs, pin_obs=True, steps=steps)
        out = self.layers[1].predict(state[1])
        return out, state