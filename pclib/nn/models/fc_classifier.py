from pclib.nn.layers import FC, FCPW, FCLI, FCSym
from pclib.utils.functional import format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCClassifier(nn.Module):
    """
    | The Standard PC Model which stacks FC layers.
    | Predictions flow from targets to inputs (top-down).
    | Heavily customisable, however, the default settings usually give best results.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)

    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__()

        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.precision_weighted = precision_weighted
        self.gamma = gamma
        self.steps = steps
        self.device = device

        self.init_layers()


    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        """
        layers = []
        in_features = None
        for out_features in [self.input_size] + self.hidden_sizes + [self.num_classes]:
            if self.precision_weighted:
                layers.append(FCPW(in_features, out_features, device=self.device, **self.factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
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
        # Reduce units
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
        if y is not None:
            state[-1]['x'] = y.clone()

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
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update top e (no prediction)
                layer.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])

        # Update Xs
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                e_below = state[i-1]['e'] if i > 0 else None
                layer.update_x(state[i], e_below, temp=temp)
        
        self.pin(state, obs, y)


    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs dependant using either y or obs if provided.
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
                if i > 0:
                    state[i-1]['x'] = layer.predict(state[i])
            if obs is not None:
                state[0]['x'] = obs.clone()
        elif obs is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = obs.clone()
                else:
                    state[i]['x'] = layer.propagate(state[i-1]['x'])

    def init_state(self, obs=None, y=None):
        """
        | Initialises the state of the model. Xs are calculated using _init_xs().

        Args:
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data

        Returns:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        with torch.no_grad():
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
        | Gets the output of the model.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)

        Returns:
            | out (torch.Tensor): Output of the model
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
        | Performs inference for the model.
        
        Args:
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Output of the model
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

    def assert_grads(model, state):
        """
        | Uses assertions to compare current gradients of each layer to manually calculated gradients.
        | Only useful if using autograd=True in training, otherwise comparing manual grads to themselves.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        """
        for i, layer in enumerate(model.layers):
            if i > 0:
                layer.assert_grad(state[i], state[i-1]['e'])                
    
    def generate(self, y, steps=None):
        """
        | Performs inference with target pinned and input free to relax.
        | In theory, should generate an input from a target.

        Args:
            | y (torch.Tensor): Target data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.
        
        Returns:
            | out (torch.Tensor): Generated input
        """
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']
    
    def classify(self, obs, steps=None):
        """
        | Performs inference on the obs once with each possible target pinned.
        | Returns the target with the lowest VFE.
        """
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"
    
        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = self.vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)

class FCClassifierLI(FCClassifier):
    """
    | This model uses FCLI layers instead of FC layers to implement lateral inhibition.
    | This introduction of competition is hoped to reduce inter-unit correlations and improve performance.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC) (NOT IMPLEMENTED)
    """
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Uses FCLI layers instead of FC layers.
        """
        layers = []
        in_features = None
        for out_features in [self.input_size] + self.hidden_sizes + [self.num_classes]:
            if self.precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCClassifierLI")
            else:
                layers.append(FCLI(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierSS(FCClassifier):
    """
    | A Self-Supervised version of FCClassifier.
    | It learns a feature extractor (self.layers) only from observations.
    | It separately learns a classifier (self.classifier) which takes the output of self.layers as input.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC) (NOT IMPLEMENTED)
        | classifier (torch.nn.Sequential): Classifier to use
    """
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        for out_features in [self.input_size] + self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError
                # layers.append(FCPW(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
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
        Takes the output of the last layer from the feature extractor and passes it through the classifier.
        Returns the output of the classifier.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e'

        Returns:
            | out (torch.Tensor): Output of the classifier
        """
        x = state[-1]['x']
        out = self.classifier(x.detach())
        return out


    def forward(self, obs=None, steps=None):
        """
        | Performs inference for the model. 
        | Uses self.classifier to get output.

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


    def classify(self, obs, steps=None):
        """
        | Performs inference on the observation and passes the output through the classifier.
        | Returns the argmax of the classifier output.

        Args:
            | obs (torch.Tensor): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Argmax(dim=1) output of the classifier
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


# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierSSLI(FCClassifierSS):
    """
    | Inherits most functionality from FCClassifierSS, so is self-supervised.
    | However, it uses FCLI layer instead of FC layers to implement lateral inhibition.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC) (NOT IMPLEMENTED)
        | classifier (torch.nn.Sequential): Classifier to use
    """

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Uses FCLI layers instead of FC layers.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        layers.append(FC(None, self.input_size, device=self.device, **self.factory_kwargs))
        in_features = self.input_size
        for out_features in self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCClassifierSSLI")
            else:
                layers.append(FCLI(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
        )

# Based on Whittington and Bogacz 2017
class FCClassifierInv(FCClassifier):
    """
    | Inherits most functionality from FCClassifier, except it predicts targets from inputs (inputs at top, targets at bottom).
    | Based on Whittington and Bogacz 2017.
    | This model is the most effective classifier, acheiving >98% val accuracy on MNIST.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
    """

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Almost identical to FCClassifier, except it swaps the position of input_size and num_classes.
        """
        layers = []
        in_features = None
        for out_features in [self.num_classes] + self.hidden_sizes + [self.input_size]:
            if self.precision_weighted:
                layers.append(FCPW(in_features, out_features, device=self.device, **self.factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

    def pin(self, state, obs=None, y=None):
        """
        | Pins the input and/or target to the state if provided.
        | Overrides FCClassifier.pin() to pin targets at bottom, and inputs at top.
        """
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()

    def _init_xs(self, state, obs=None, y=None):
        """
        | Initialises xs dependant using either obs or y if provided.    
        | Similar to FCClassifier._init_xs(), but generates predictions from obs, or propagates error from target.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
            | obs (Optional[torch.Tensor]): Input data
            | y (Optional[torch.Tensor]): Target data
        """
        if obs is not None:
            for i, layer in reversed (list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = y.clone()
                else:
                    state[i]['x'] = layer.propagate(state[i-1]['x'])

    def get_output(self, state):
        """
        | Gets the output of the model.
        | Output is pulled from bottom layer, instead of top layer.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e' (and 'eps' for FCPW)
        
        Returns:
            | out (torch.Tensor): Output of the model
        """
        return state[0]['x']

class FCClassifierInvSym(FCClassifierInv):
    """
    | Inherits most functionality from FCClassifierInv, except it uses FCSym layers instead of FC layers.
    | In theory is functionally symmetrical. Predictions flow both ways, and same for error propagations.
    | Currently does not learn, but may just need more hyperparameter tuning.
    
    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
    """
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Identical to FCClassifierInv, except it uses FCSym layers instead of FC layers.
        """
        layers = []
        features = [self.num_classes] + self.hidden_sizes + [self.input_size]
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

class FCClassifierInvSS(FCClassifierInv):
    """
    | Inherits most functionality from FCClassifierInv, except it is self-supervised.
    | It learns a feature extractor (self.layers) only from observations.
    | It separately learns a classifier (self.classifier) which takes the output of self.layers as input.

    Args:
        | input_size (int): Number of input features
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
        | input_size (int): Number of input features
        | num_classes (int): Number of classes
        | hidden_sizes (list): List of hidden layer sizes
        | steps (int): Number of steps to run inference for
        | bias (bool): Whether to include bias in layers
        | symmetric (bool): Whether to use same weights for top-down prediction and bottom-up error prop.
        | precision_weighted (bool): Whether to use precision weighted layers (FCPW instead of FC)
        | classifier (torch.nn.Sequential): Classifier to use
    """
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.tanh, d_actv_fn=None, gamma=0.1, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, device, dtype)

    def init_layers(self):
        """
        | Initialises self.layers based on input parameters.
        | Doesn't include num_classes layer in self.layers.
        | Initialises self.classifier as a simple MLP, to classify the output of self.layers.
        """
        layers = []
        in_features = None
        for out_features in self.hidden_sizes + [self.input_size]:
            if self.precision_weighted:
                layers.append(FCPW(in_features, out_features, device=self.device, **self.factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], 200, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
            nn.ReLU(),
            nn.Linear(200, self.num_classes, bias=True, device=self.device, dtype=self.factory_kwargs['dtype']),
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
        | Takes the output from the feature extractor (bottom layer) and passes it through the classifier.

        Args:
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'

        Returns:
            | out (torch.Tensor): Output of the classifier
        """
        x = state[0]['x']
        out = self.classifier(x.detach())
        return out

    def forward(self, obs=None, steps=None):
        """
        | Performs inference for the model.
        | does not pin targets, so is self-supervised.
        | Uses self.classifier to get output.

        Args:
            | obs (Optional[torch.Tensor]): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Output of the model
            | state (list): List of layer state dicts, each containing 'x' and 'e_l' and 'e_u'
        """
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp=temp)
            
        out = self.get_output(state)
            
        return out, state
    
    def classify(self, obs, steps=None):
        """
        | Performs inference on the observation and passes the output through the classifier.
        | Returns the argmax of the classifier output.

        Args:
            | obs (torch.Tensor): Input data
            | steps (Optional[int]): Number of steps to run inference for. Uses self.steps if not provided.

        Returns:
            | out (torch.Tensor): Argmax(dim=1) output of the classifier
        """
        return self.forward(obs, steps)[0].argmax(dim=1)
    
    def generate(self, y, steps=None):
        """
        | Not implemented as one cannot generate an input without a target, and this model does not pin targets.
        """
        raise NotImplementedError("Generate not implemented for FCClassifierInvSS")