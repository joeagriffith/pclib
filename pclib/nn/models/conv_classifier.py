from pclib.nn.layers import Conv2d, FC
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight
from typing import List, Optional

# Based on Whittington and Bogacz 2017
class ConvClassifier(nn.Module):
    """
    | Similar to the FCClassifier, except uses convolutions instead of fully connected layers.
    | This network is not currently customisable, but requires altering the init_layers() code to change the architecture.

    Parameters
    ----------
        steps : int 
            Number of steps to run the network for in inference phase.
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
        has_top: bool
            Whether to include a recurrent layer for top prediction
        device : torch.device
            Device to run the network on.
        dtype : torch.dtype
            Data type to use for network parameters.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(
            self, 
            steps:int = 20, 
            bias:bool = True, 
            symmetric:bool = True, 
            actv_fn:callable = F.relu, 
            d_actv_fn:callable = None, 
            gamma:float = 0.1, 
            x_decay:float = 0.0,
            has_top:bool = False,
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'x_decay': x_decay, 'dtype': dtype}
        super().__init__()

        self.num_classes = 10
        self.steps = steps
        self.bias = bias
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.x_decay = x_decay
        self.device = device
        self.dtype = dtype

        self.init_layers()
        self.register_buffer('epochs_trained', torch.tensor(0, dtype=torch.long))
        self.register_buffer('max_vfe', torch.tensor(-float('inf'), dtype=torch.float32))


    def __str__(self):
        """
        | Returns a string representation of the Model.

        Returns
        -------
            str
        """
        base_str = super().__str__()

        custom_info = "\n  (params): \n" + \
            f"    in_shape: (-1, 1, 32, 32)" + \
            f"    out_shape: (-1, 10)" + \
            f"    steps: {self.steps}" + \
            f"    bias: {self.factory_kwargs['has_bias']}" + \
            f"    symmetric: {self.factory_kwargs['symmetric']}" + \
            f"    actv_fn: {self.factory_kwargs['actv_fn'].__name__}" + \
            f"    gamma: {self.factory_kwargs['gamma']}" + \
            f"    x_decay: {self.factory_kwargs['x_decay']}" + \
            f"    has_top: {self.has_top}" + \
            f"    device: {self.device}" + \
            f"    dtype: {self.factory_kwargs['dtype']}" + \
            f"    epochs_trained: {self.epochs_trained}" + \
            f"    max_vfe: {self.max_vfe}\n"
        
        string = base_str[:base_str.find('\n')] + custom_info + base_str[base_str.find('\n'):]
        
        return string


    def init_layers(self):
        """
        | Initialises the layers of the network.
        | Not currently customisable, but can be changed by altering this code.
        """
        layers = []
        layers.append(Conv2d(None, (1, 32, 32),         maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((1, 32, 32), (32, 16, 16), maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((32, 16, 16), (64, 8, 8),  maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((64, 8, 8), (128, 4, 4),    maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((128, 4, 4), (256, 2, 2),    maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((256, 2, 2), (256, 1, 1),    maxpool=2, **self.factory_kwargs))
        layers.append(FC(256, 128, **self.factory_kwargs))
        layers.append(FC(128, 10, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

        if self.has_top:
            self.top = FC(
                10, 
                10, 
                has_bias=self.bias,
                actv_fn=self.actv_fn,
                gamma=self.gamma,
                x_decay=self.x_decay,
                device=self.device,
                dtype=self.dtype,
            )
            self.top.weight.data = (self.top.weight.data - torch.diag(torch.diag(self.top.weight.data))) * 0.01

    def inc_epochs(self, n:int=1):
        """
        | Increments the number of epochs trained by n.

        Parameters
        ----------
            n : int
                Number of epochs to increment by
        """
        self.epochs_trained += n


    def vfe(self, state:List[dict], batch_reduction:str='mean', normalise:bool=False):
        """
        | Calculates the Variational Free Energy (VFE) of the model.
        | This is the sum of the squared prediction errors of each layer.
        | how batches and units are reduced is controlled by batch_reduction and unit_reduction.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
            batch_reduction : str
                How to reduce over batches ['sum', 'mean', None]
            unit_reduction : str
                How to reduce over units ['sum', 'mean']
            normalise : bool
                Whether to normalise the VFE by the number of units.

        Returns
        -------
            torch.Tensor
                VFE of the model (scalar)
        """
        # Reduce units for each layer
        vfe = [0.5 * state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        
        if normalise:
            vfe = [vfe_i / (vfe_i.detach() + 1e-6) for vfe_i in vfe]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    # def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None):
    #     """
    #     | Initialises Xs.
    #     | If y is provided, xs are initialised top-down using predictions.
    #     | Else if obs is provided, xs are initialised bottom-up using propagations.
        
    #     Parameters
    #     ----------
    #         state : List[dict]
    #             List of state dicts for each layer, each containing 'x' and 'e' tensors.
    #         obs : Optional[torch.Tensor]
    #             Input data
    #         y : Optional[torch.Tensor]
    #             Target data
    #     """
    #     with torch.no_grad():
    #         if y is not None:
    #             for i, layer in reversed(list(enumerate(self.layers))):
    #                 if i == len(self.layers) - 1: # last layer
    #                     state[i]['x'] = y.detach()
    #                 if i > 0:
    #                     pred = layer.predict(state[i])
    #                     if isinstance(layer, FC) and isinstance(self.layers[i-1], Conv2d):
    #                         shape = self.layers[i-1].shape
    #                         pred = pred.view(pred.shape[0], shape[0], shape[1], shape[2])
    #                     state[i-1]['x'] = pred.detach()
    #             if obs is not None:
    #                 state[0]['x'] = obs.detach()

    #         elif obs is not None:
    #             for i, layer in enumerate(self.layers):
    #                 if i == 0:
    #                     state[0]['x'] = obs.detach()
    #                 else:
    #                     x_below = state[i-1]['x'].detach()
    #                     state[i]['x'] = layer.propagate(x_below)

    # def _init_es(self, state:List[dict]):
    #     """
    #     | Calculates the initial errors for each layer.
    #     | Assumes that Xs have already been initialised.

    #     Parameters
    #     ----------
    #         state : List[dict]
    #             List of state dicts for each layer, each containing 'x' and 'e' tensors.
    #     """
    #     with torch.no_grad():
    #         for i, layer in enumerate(self.layers):
    #             if i < len(self.layers) - 1:
    #                 pred = self.layers[i+1].predict(state[i+1])
    #                 layer.update_e(state[i], pred)

    def init_state(self, obs:torch.Tensor = None, y:torch.Tensor = None):
        """
        | Initialises the state of the network.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data

        Returns
        -------
            List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
        """
        if obs is not None:
            b_size = obs.shape[0]
        elif y is not None:
            b_size = y.shape[0]
        else:
            raise ValueError('Either obs or y must be provided to init_state.')
        state = []
        for layer in self.layers:
            state.append(layer.init_state(b_size))

        if obs is not None:
            state[0]['x'] = obs.clone()
        if y is not None:
            state[-1]['x'] = y.clone()
        
        # # Alternative initialisation
        # self._init_xs(state, obs, y)
        # self._init_es(state)
        
        return state

    def to(self, device):
        self.device = device
        for layer in self.layers:
            layer.to(device)
        return self

    def get_output(self, state:List[dict]):
        """
        | Returns the output of the network.

        Parameters
        ----------
            state : List[dict]
                List of layer state dicts, each containing 'x' and 'e'

        Returns
        -------
            torch.Tensor
                Output of the network
        """
        return state[-1]['x']

    def step(self, state:List[dict], gamma:torch.Tensor=None, pin_obs:bool = False, pin_target:bool = False):
        """
        Performs one step of inference, updating all Xs first, then calculates Errors.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            temp : Optional[float]
                Temperature for simulated annealing
            gamma : Optional[float]
                Step size for x updates
        """

        for i, layer in enumerate(self.layers):
            if i > 0 or not pin_obs:
                if i < len(self.layers) - 1 or not pin_target:
                    e_below = state[i-1]['e'] if i > 0 else None
                    layer.update_x(state[i], e_below, gamma=gamma)
        
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
            elif self.has_top:
                pred = self.top.predict(state[-1])
            else:
                continue
            layer.update_e(state[i], pred)

    def update_gamma(self, state, gamma, prev_vfe=None):
        """
        | Decays gamma based on the change in VFE.
        | If VFE increases, gamma is decayed, else it is increased.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
            gamma : torch.Tensor
                Current gamma
            prev_vfe : Optional[torch.Tensor]
                VFE from the previous step

        Returns
        -------
            torch.Tensor
                Updated gamma
            torch.Tensor
                Current VFE
        """
        vfe = self.vfe(state, batch_reduction=None)
        if prev_vfe is None:
            return gamma, vfe
        else:
            mult = torch.where(vfe < prev_vfe, 1.0, 0.9)
            return gamma * mult, vfe

    def forward(self, obs:torch.Tensor = None, pin_obs:bool = False, steps:int = None):
        """
        | Performs inference for the network.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            steps : Optional[int]
                Number of steps to run inference for
            learn_on_step : bool
                Whether to backpropagate on each step. Default False.
        
        Returns
        -------
            torch.Tensor
                Output of the network
            List[dict]
                List of layer state dicts, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        prev_vfe = None
        gamma = torch.ones(state[0]['x'].shape[0]).to(self.device) * self.gamma
        for i in range(steps):
            self.step(state, gamma, pin_obs)
            with torch.no_grad():
                gamma, prev_vfe = self.update_gamma(state, gamma, prev_vfe)
            
        out = self.get_output(state)
            
        return out, state

    def classify(self, obs:torch.Tensor, state:List[dict] = None, steps:int=None):
        """
        | Classifies the input obs.

        Parameters
        ----------
            obs : torch.Tensor
                Input data
            state : Optional[List[dict]]
                List of layer state dicts, each containing 'x' and 'e'
            steps : Optional[int]
                Number of steps to run inference for
        
        Returns
        -------
            torch.Tensor
                Predicted class
        """
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, pin_obs=True, pin_target=True, steps=steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)