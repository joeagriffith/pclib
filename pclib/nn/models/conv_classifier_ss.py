import torch
import torch.nn as nn
import torch.nn.functional as F

from pclib.nn.layers import ConvTranspose2d, Conv2d
from pclib.nn.models import ConvClassifier

class ConvClassifierSS(ConvClassifier):
    """
    | Similar to ConvClassifier, but only uses PC for convolutation Feature Extractor
    | Trains feature extractor unsupervised, then trains classifier supervised.

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
    def __init__(self, steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(steps=steps, bias=bias, symmetric=symmetric, precision_weighted=precision_weighted, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, device=device, dtype=dtype)

    def init_layers(self):
        """
        | Initialises self.layers with the layers of the network.
        """
        layers = []
        layers.append(ConvTranspose2d((1, 28, 28), None, **self.factory_kwargs))
        layers.append(ConvTranspose2d((32, 24, 24), 1, 5, padding=0, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 10, 10), 32, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 3, 3), 64, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(ConvTranspose2d((128, 1, 1), 64, 3, padding=0, **self.factory_kwargs))

        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 200, bias=True, device=self.device),
            nn.ReLU(),
            nn.Linear(200, 10, bias=True, device=self.device),
        )

    def to(self, device):
        """
        | Moves the network to the device.

        Args:
            | device (torch.device): Device to move the network to.
        """
        self.device = device
        for layer in self.layers:
            layer.to(device)
        for layer in self.classifier:
            layer.to(device)
        return self
    
    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialise xs with gaussian noise.
        | Cannot initialise from y as model is unsupervised
        """
        y = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                state[0]['x'] = obs.clone()
            else:
                state[i]['x'] = 0.01 * torch.randn_like(state[i]['x']).to(self.device)
        for i, layer in enumerate(self.layers):
            if i == 0 and obs is not None:
                continue
            state[i]['x'].requires_grad = True

        self.pin(state, obs, y)

    def get_output(self, state):
        """
        | Returns the output of the network.
        | Does this by passing the top layer through the classifier.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns:
            | out (torch.Tensor): Output of the network
        """
        return self.classifier(state[-1]['x'].detach())


    def forward(self, obs=None, steps=None):
        """
        | Performs inference for the model.
        | uses self.classifier to get output.
        | Doesn't pin targets, so is self-supervised.

        Args:
            | obs (Optional[torch.Tensor]): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Output of the model
            | state (list): List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp=temp)

        out = self.get_output(state)

        return out, state

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
        
        return state[0]['x'], state
    
    def generate(self, y, steps=None):
        """
        | Generates an image from the target y.
        | Not implemented, as model is unsupervised.
        """
        raise NotImplementedError

    def classify(self, obs, state=None, steps=None):
        return self.forward(obs)[0].argmax(dim=1)