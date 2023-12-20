import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pclib.nn.layers import FC, Conv2d, ConvTranspose2d
from pclib.nn.models import ConvClassifier
from pclib.utils.functional import format_y

class ConvClassifierInv(ConvClassifier):
    """
    | Similar to ConvClassifier, but uses convolutions in the inverse direction.
    | Predictions flow from observation to target, and errors propagate from target to observation.

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
    def __init__(self, steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__()

    def init_layers(self):
        """
        | Initialises self.layers with the layers of the network.
        | Puts FC layers before Conv layers, and uses Conv2d instead of ConvTranspose2d.
        """
        layers = []
        layers.append(FC(None, 10, **self.factory_kwargs))
        layers.append(FC(10, 3*3*64, **self.factory_kwargs))
        layers.append(Conv2d((32, 10, 10), 64, 5, padding=0, maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((32, 24, 24), 32, 5, padding=0, maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((1, 28, 28), 32, 5, padding=0, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

    def pin(self, state, obs=None, y=None): 
        """
        | Pins the input and/or target to the state if provided.
        | Not permanent, so is called after every x update.
        | obs is pinned to top layer, y is pinned to bottom layer.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()

    def step(self, state, obs=None, y=None, temp=None):
        """
        | Performs on step of inference, updating Es first, then calculates X updates using .backward() on the VFE.
        | Es are updated top-down, collecting predictions along the way, then Xs are updated bottom-up.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | temp (Optional[float]): Temperature for the softmax function

        """
        self.zero_grad()

        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update top e (no prediction)
                layer.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])
                if isinstance(layer, FC) and isinstance(self.layers[i-1], (Conv2d, ConvTranspose2d)):
                    shape = self.layers[i-1].shape
                    pred = pred.reshape(pred.shape[0], shape[0], shape[1], shape[2])
                elif isinstance(layer, (Conv2d, ConvTranspose2d)) and isinstance(self.layers[i-1], FC):
                    pred = pred.flatten(1)

        self.vfe(state).backward(retain_graph=True)
        # for i, layer in enumerate(self.layers):
        #     if isinstance(layer, (Conv2d, ConvTranspose2d)):
        #         if i == 0 and obs is not None:
        #             continue
        #         elif i == len(self.layers) - 1 and y is not None:
        #             continue
        #         state[i]['x'].grad = -2 * (torch.)

        for i, layer in enumerate(self.layers):
            e_below = state[i-1]['e'] if i > 0 else None
            if isinstance(layer, FC) and isinstance(self.layers[i-1], (ConvTranspose2d, Conv2d)) and i > 0:
                e_below = e_below.flatten(1)
            if i == 0 and y is not None:
                continue
            elif i == len(self.layers) - 1 and obs is not None:
                continue
            layer.update_x(state[i], e_below)

        self.pin(state, obs, y)
    # def step(self, state, obs=None, y=None, temp=None):
    #     """
    #     | Performs on step of inference, updating Es first, then calculates X updates using .backward() on the VFE.
    #     | This version uses conv2d_input to calculate the X updates.

    #     Args:
    #         | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
    #         | obs (Optional[torch.Tensor]): Input data
    #         | y (Optional[torch.Tensor]): Target data
    #         | temp (Optional[float]): Temperature for the softmax function
    #     """

    #     self.zero_grad()

    #     for i, layer in reversed(list(enumerate(self.layers))):
    #         if i < len(self.layers) - 1:
    #             layer.update_e(state[i], pred, temp=temp)
    #         if i > 0:
    #             pred = layer.predict(state[i])
    #             if isinstance(layer, FC) and isinstance(self.layers[i-1], Conv2d):
    #                 shape = self.layers[i-1].shape
    #                 pred = pred.reshape(pred.shape[0], shape[0], shape[1], shape[2])
    #             elif isinstance(layer, Conv2d) and isinstance(self.layers[i-1], FC):
    #                 pred = pred.flatten(1)

    #     for i, layer in enumerate(self.layers):
    #         if isinstance(layer, Conv2d):
    #             if i == 0 and obs is not None:
    #                 continue
    #             elif i == len(self.layers) - 1 and y is not None:
    #                 continue
    #             state[i]['x'].grad = torch.zeros_like(state[i]['x'])
    #             if i > 0:
    #                 e_below = state[i-1]['e']
    #                 if e_below.dim() == 2:
    #                     e_below = e_below.reshape(e_below.shape[0], layer.prev_channels, -1)
    #                     e_below = e_below.reshape(e_below.shape[0], layer.prev_channels, int(math.sqrt(e_below.shape[-1])), int(math.sqrt(e_below.shape[-1])))
    #                 state[i]['x'].grad += -2 * torch.nn.grad.conv2d_input(state[i]['x'].shape, layer.conv_td[0].weight, e_below, layer.conv_td[0].stride, layer.conv_td[0].padding)
    #             if i < len(self.layers) - 1:
    #                 state[i]['x'].grad += -2 * state[i]['e']

    #     for i, layer in enumerate(self.layers):
    #         e_below = state[i-1]['e'] if i > 0 else None
    #         if isinstance(layer, FC) and isinstance(self.layers[i-1], Conv2d):
    #             e_below = e_below.flatten(1)
    #         if i == 0 and obs is not None:
    #             continue
    #         elif i == len(self.layers) - 1 and y is not None:
    #             continue
    #         layer.update_x(state[i], e_below)

    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs using obs if provided.
        | If obs is provided, xs are initialised top-down using predictions.
        | Xs cannot be initialised using y, as this would require propagation, which is not implemented.
        
        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    pred = layer.predict(state[i])
                    if isinstance(layer, Conv2d) and isinstance(self.layers[i-1], FC):
                        pred = pred.flatten(1)
                    state[i-1]['x'] = pred.detach()
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            raise NotImplementedError
        #     for i, layer in enumerate(self.layers):
        #         if i == 0:
        #             state[0]['x'] = y.clone()
        #         else:
        #             x_below = state[i-1]['x']
        #             if state[i]['x'].dim() == 4 and state[i-1]['x'].dim() == 2:
        #                 x_below = x_below.unsqueeze(-1).unsqueeze(-1)
        #             state[i]['x'] = layer.propagate(x_below)
        for i, layer in enumerate(self.layers):
            if i == 0 and y is not None:
                continue
            elif i == len(self.layers) - 1 and obs is not None:
                continue
            if isinstance(layer, (Conv2d, ConvTranspose2d)):
                state[i]['x'].requires_grad = True

    def get_output(self, state):
        """
        | Returns the output of the network.
        
        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns:
            | out (torch.Tensor): Output of the network
        """
        return state[0]['x']
    
    def generate(self, y, steps=None):
        """
        | Generates an image from the target y.
        | Not implemented, as this would require propagation, which is not implemented.
        """
        raise NotImplementedError

    def classify(self, obs, state=None, steps=None):
        """
        | Classifies the input obs.
        | Does this by calculating the VFE for each possible class, and choosing the class with the lowest VFE.

        Args:
            | obs (torch.Tensor): Input data
            | state (Optional[list]): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | steps (Optional[int]): Number of steps to run inference for

        Returns:
            | out (torch.Tensor): Predicted class
        """
        if steps is None:
            steps = self.steps


        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)