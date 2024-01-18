from pclib.nn.layers import Conv2d, FC
from pclib.nn.models import ConvClassifier
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

# Based on Whittington and Bogacz 2017
class ConvClassifierUs(ConvClassifier):
    """
    | Similar to the FCClassifier, except uses convolutions instead of fully connected layers.
    | Currently calculates X updates using .backward() on the VFE, which is slow.
    | May be possible to speed up by calculating X updates manually, but requires complex indexing to minimise floating point operations.
    | This network is not currently customisable, but requires altering the init_layers() code to change the architecture.

    Args:
        | steps (int): Number of steps to run the network for.
        | bias (bool): Whether to include bias terms in the network.
        | symmetric (bool): Whether to use symmetric weights. 
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
    def __init__(self, steps=20, bias=True, symmetric=True, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(
            steps=steps,
            bias=bias,
            symmetric=symmetric,
            actv_fn=actv_fn,
            d_actv_fn=d_actv_fn,
            gamma=gamma,
            device=device,
            dtype=dtype,
        )

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(Conv2d(None,          (1, 32, 32),                  **self.factory_kwargs))
        layers.append(Conv2d((1, 32, 32),   (32, 16, 16),  5, 2, 2, **self.factory_kwargs))
        layers.append(Conv2d((32, 16, 16),  (64, 8, 8),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((64, 8, 8),    (128, 4, 4),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((128, 4, 4),    (256, 2, 2),    3, 2, 1, **self.factory_kwargs))
        layers.append(Conv2d((256, 2, 2),    (256, 1, 1),    3, 2, 1, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

        # layers = []
        # layers.append(Conv2d(None,         (1, 28, 28),                  **self.factory_kwargs))
        # layers.append(Conv2d((1, 28, 28),  (32, 22, 22), 7, 1, 0, **self.factory_kwargs))
        # layers.append(Conv2d((32, 22, 22), (64, 16, 16), 7, 1, 0, **self.factory_kwargs))
        # layers.append(Conv2d((64, 16, 16), (128, 10, 10), 7, 1, 0, **self.factory_kwargs))
        # layers.append(Conv2d((128, 10, 10), (128, 4, 4), 7, 1, 0, **self.factory_kwargs))
        # layers.append(Conv2d((128, 4, 4), (128, 1, 1), 4, 1, 0, **self.factory_kwargs))
        # self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.layers[-1].shape[0], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=False, device=self.device, dtype=self.factory_kwargs['dtype']),
        )


    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        for layer in self.classifier:
            layer.to(device)
        return self

    def get_output(self, state):
        """
        | Returns the output of the network.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns:
            | out (torch.Tensor): Output of the network
        """
        x = state[-1]['x']
        out = self.classifier(x.detach().flatten(1))
        return out
        

    def forward(self, obs=None, steps=None):
        """
        | Performs inference for the network.

        Args:
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | steps (Optional[int]): Number of steps to run inference for
        
        Returns:
            | out (torch.Tensor): Output of the network
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp)
            
        out = self.get_output(state)
            
        return out, state

    def classify(self, obs, steps=None):
        """
        | Classifies the input obs.

        Args:
            | obs (torch.Tensor): Input data
            | steps (Optional[int]): Number of steps to run inference for
        
        Returns:
            | out (torch.Tensor): Predicted class
        """
        return self.forward(obs, steps)[0].argmax(dim=1)


    def reconstruct(self, obs, steps=None):
        """
        | Initialises the state of the model using the observation.
        | Runs inference without pinning the observation.
        | In theory should reconstruct the observation.

        Args:
            | obs (torch.Tensor): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Reconstructed observation
            | state (list): List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp)
        
        out = state[0]['x']

        return out, state

    
    def generate(self, y, steps=None):
        """
        | Not implemented as one cannot generate an input without a target, and this model does not pin targets.
        """
        raise(NotImplementedError)

