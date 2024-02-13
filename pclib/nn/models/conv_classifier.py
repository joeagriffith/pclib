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
        temp_k : float
            Temperature constant for inference
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
            temp_k:float = 1.0, 
            device:torch.device = torch.device('cpu'), 
            dtype:torch.dtype = None
        ):
        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        super().__init__()

        self.num_classes = 10
        self.steps = steps
        self.gamma = gamma
        self.temp_k = temp_k
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
        layers.append(Conv2d((64, 8, 8), (64, 4, 4),    maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((64, 4, 4), (64, 2, 2),    maxpool=2, **self.factory_kwargs))
        layers.append(Conv2d((64, 2, 2), (64, 1, 1),    maxpool=2, **self.factory_kwargs))
        layers.append(FC(64, 128, **self.factory_kwargs))
        layers.append(FC(128, 10, **self.factory_kwargs))
        self.layers = nn.ModuleList(layers)

    def inc_epochs(self, n:int=1):
        """
        | Increments the number of epochs trained by n.

        Parameters
        ----------
            n : int
                Number of epochs to increment by
        """
        self.epochs_trained += n


    def vfe(self, state:List[dict], batch_reduction:str='mean', unit_reduction:str='sum'):
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

        Returns
        -------
            torch.Tensor
                VFE of the model (scalar)
        """
        # Reduce units for each layer
        if unit_reduction == 'sum':
            vfe = [-0.5 * state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        elif unit_reduction =='mean':
            vfe = [-0.5 * state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state]
        # Reduce layers
        vfe = sum(vfe)
        # Reduce batches
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    def _init_xs(self, state:List[dict], obs:torch.Tensor = None, y:torch.Tensor = None):
        """
        | Initialises Xs.
        | If y is provided, xs are initialised top-down using predictions.
        | Else if obs is provided, xs are initialised bottom-up using propagations.
        
        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
        """
        with torch.no_grad():
            if y is not None:
                for i, layer in reversed(list(enumerate(self.layers))):
                    if i == len(self.layers) - 1: # last layer
                        state[i]['x'] = y.detach()
                    if i > 0:
                        pred = layer.predict(state[i])
                        if isinstance(layer, FC) and isinstance(self.layers[i-1], Conv2d):
                            shape = self.layers[i-1].shape
                            pred = pred.view(pred.shape[0], shape[0], shape[1], shape[2])
                        state[i-1]['x'] = pred.detach()
                if obs is not None:
                    state[0]['x'] = obs.detach()

            elif obs is not None:
                for i, layer in enumerate(self.layers):
                    if i == 0:
                        state[0]['x'] = obs.detach()
                    else:
                        x_below = state[i-1]['x'].detach()
                        state[i]['x'] = layer.propagate(x_below)

    def _init_es(self, state:List[dict]):
        """
        | Calculates the initial errors for each layer.
        | Assumes that Xs have already been initialised.

        Parameters
        ----------
            state : List[dict]
                List of state dicts for each layer, each containing 'x' and 'e' tensors.
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i < len(self.layers) - 1:
                    pred = self.layers[i+1].predict(state[i+1])
                    layer.update_e(state[i], pred, temp=1.0)

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
        
        self._init_xs(state, obs, y)
        self._init_es(state)
        
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


    def calc_temp(self, step_i:int, steps:int):
        """
        | Calculates the temperature for the current step.
        | Uses a geometric sequence to decay the temperature.
        | temp = self.temp_k * \alpha^{step_i}
        | where \alpha = (\frac{0.001}{1})^{\frac{1}{steps}}

        Parameters
        ----------
            step_i : int
                Current step
            steps : int
                Total number of steps

        Returns
        -------
            float
                Temperature for the current step
        """
        alpha = (0.001/1)**(1/steps)
        return self.temp_k * alpha**step_i


    def step(self, state:List[dict], obs:torch.Tensor=None, y:torch.Tensor=None, temp:float=None, gamma:float=None):
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
            if i > 0 or obs is None:
                if i < len(self.layers) - 1 or y is None:
                    e_below = state[i-1]['e'] if i > 0 else None
                    layer.update_x(state[i], e_below, temp=temp, gamma=gamma)
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                pred = self.layers[i+1].predict(state[i+1])
                layer.update_e(state[i], pred, temp=temp)


    def forward(self, obs:torch.Tensor = None, y:torch.Tensor = None, steps:int = None, learn_on_step:bool = False):
        """
        | Performs inference phase of the network.

        Parameters
        ----------
            obs : Optional[torch.Tensor]
                Input data
            y : Optional[torch.Tensor]
                Target data
            steps : Optional[int]
                Number of steps to run inference for
            learn_on_step : bool
                Whether to backpropagate on each step. Default False.
        
        Returns
        -------
            torch.Tensor
                Output of the network
            List[dict]
                List of state dicts for each layer, each containing 'x' and 'e'
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, y)
    
        prev_vfe = None
        gamma = self.gamma
        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, y, temp, gamma)
            vfe = self.vfe(state)
            if learn_on_step:
                vfe.backward()
            if prev_vfe is not None and vfe < prev_vfe:
                gamma = gamma * 0.9
            prev_vfe = vfe
            
        out = self.get_output(state)
            
        return out, state

    def generate(self, y:torch.Tensor, steps:int = None):
        """
        | Generates an image from the target y.

        Parameters
        ----------
            y : torch.Tensor
                Target data
            steps : Optional[int]
                Number of steps to run inference for. defaults to self.steps

        Returns
        -------
            torch.Tensor
                Generated image
        """
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']
    
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
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)