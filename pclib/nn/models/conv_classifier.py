from pclib.nn.layers import Conv2d, ConvTranspose2d, FC
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

# Based on Whittington and Bogacz 2017
class ConvClassifier(nn.Module):
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
        super().__init__()

        self.num_classes = 10
        self.precision_weighted = precision_weighted
        self.steps = steps
        self.device = device

        self.init_layers()
        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('min_vfe', torch.tensor(float('inf'), dtype=torch.float32))

    def inc_epochs(self, n=1):
        """
        | Increments the number of epochs trained by n.

        Args:
            | n (int): Number of epochs to increment by
        """
        self.epochs_trained += n

    def init_layers(self):
        """
        | Initialises the layers of the network.
        """
        layers = []
        layers.append(ConvTranspose2d((1, 28, 28), None, **self.factory_kwargs))
        layers.append(ConvTranspose2d((32, 24, 24), 1, 5, padding=0, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 10, 10), 32, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(ConvTranspose2d((64, 3, 3), 64, 5, padding=0, upsample=2, **self.factory_kwargs))
        layers.append(FC(64*3*3, 128, **self.factory_kwargs))
        layers.append(FC(128, 10, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

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
        if unit_reduction == 'sum':
            vfe = [state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        elif unit_reduction =='mean':
            vfe = [state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    def pin(self, state, obs=None, y=None):
        """
        | Pins the input and/or target to the state if provided.
        | Not permanent, so is called after every x update.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data

        """
        if obs is not None:
            state[0]['x'] = obs.clone()
            if isinstance(self.layers[0], (ConvTranspose2d, Conv2d)):
                state[0]['x'].requires_grad = True
        if y is not None:
            state[-1]['x'] = y.clone()
            if isinstance(self.layers[-1], (ConvTranspose2d, Conv2d)):
                state[-1]['x'].requires_grad = True

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
            if i == 0 and obs is not None:
                continue
            elif i == len(self.layers) - 1 and y is not None:
                continue
            layer.update_x(state[i], e_below)

        self.pin(state, obs, y)

    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs using y if provided.
        | If y is provided, xs are initialised top-down using predictions.
        | Else if just obs is provided, xs can't be initialised, so are initialised randomly.
        
        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if y is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = y.clone()
                if i > 0:
                    pred = layer.predict(state[i])
                    if isinstance(layer, FC) and isinstance(self.layers[i-1], (ConvTranspose2d, Conv2d)):
                        shape = self.layers[i-1].shape
                        pred = pred.reshape(pred.shape[0], shape[0], shape[1], shape[2])
                    state[i-1]['x'] = pred.detach()
            if obs is not None:
                state[0]['x'] = obs.clone()

        elif obs is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = obs.clone()
                else:
                    state[i]['x'] = 0.01 * torch.randn_like(state[i]['x']).to(self.device)

        # elif obs is not None:
        #     raise(NotImplementedError, "Initialising xs from obs not implemented, because propagate dont work.")
        #     for i, layer in enumerate(self.layers):
        #         if i == 0:
        #             state[0]['x'] = obs.clone()
        #         else:
        #             x_below = state[i-1]['x']
        #             if isinstance(layer, FC) and isinstance(self.layers[i-1], ConvTranspose2d):
        #                 x_below = x_below.flatten(1)
        #             state[i]['x'] = layer.propagate(x_below)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2d, ConvTranspose2d)):
                state[i]['x'].requires_grad = True

    def init_state(self, obs=None, y=None):
        """
        | Initialises the state of the network.

        Args:
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data

        Returns:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        if obs is not None:
            b_size = obs.shape[0]
        elif y is not None:
            b_size = y.shape[0]
        state = []
        for layer in self.layers:
            state.append(layer.init_state(b_size))
        
        self._init_xs(state, obs, y)
        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
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
        return state[-1]['x']

    def calc_temp(self, step_i, steps):
        """
        | Calculates the temperature for the current step.

        Args:
            | step_i (int): Current step
            | steps (int): Total number of steps
        
        Returns:
            | temp (float): Temperature for the current step
        """
        return 1 - (step_i / steps)

    def forward(self, obs=None, y=None, steps=None):
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

        state = self.init_state(obs, y)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, y, temp)
            
        out = self.get_output(state)
            
        return out, state

    def generate(self, y, steps=None):
        """
        | Generates an image from the target y.

        Args:
            | y (torch.Tensor): Target data
            | steps (Optional[int]): Number of steps to run inference for

        Returns:
            | out (torch.Tensor): Generated image
        """
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']
    
    def classify(self, obs, state=None, steps=None):
        """
        | Classifies the input obs.

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



