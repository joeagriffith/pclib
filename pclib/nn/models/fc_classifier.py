from pclib.nn.layers import FC, FCPW, FCLI, FCSym
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifier(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__()

        self.factory_kwargs = {'actv_fn': actv_fn, 'd_actv_fn': d_actv_fn, 'gamma': gamma, 'beta': beta, 'has_bias': bias, 'symmetric': symmetric, 'dtype': dtype}
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.precision_weighted = precision_weighted
        self.gamma = gamma
        self.beta = beta
        self.steps = steps
        self.device = device

        self.init_layers()


    def init_layers(self):
        layers = []
        in_features = None
        for out_features in [self.input_size] + self.hidden_sizes + [self.num_classes]:
            if self.precision_weighted:
                layers.append(FCPW(in_features, out_features, device=self.device, **self.factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, device=self.device, **self.factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)
    

    def vfe(self, state, batch_reduction='mean', layer_reduction='sum'):
        if layer_reduction == 'sum':
            vfe = sum([state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
        elif layer_reduction =='mean':
            vfe = sum([state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
        if batch_reduction == 'sum':
            vfe = vfe.sum()
        elif batch_reduction == 'mean':
            vfe = vfe.mean()

        return vfe

    # Pin input and output Xs if provided
    def pin(self, state, obs=None, y=None):
        if obs is not None:
            state[0]['x'] = obs.clone()
        if y is not None:
            state[-1]['x'] = y.clone()

    def step(self, state, obs=None, y=None, temp=None):

        # Update Es, Top-down so we can collect predictions as we descend
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < len(self.layers) - 1: # don't update top e (no prediction)
                layer.update_e(state[i], pred, temp=temp)
            if i > 0: # Bottom layer can't predict
                pred = layer.predict(state[i])

        # Update Xs
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                e_below = state[i-1]['e'] if i > 0 else None
                layer.update_x(state[i], e_below)
        
        self.pin(state, obs, y)


    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
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
        return state[-1]['x']

    def calc_temp(self, step_i, steps):
        return 1 - (step_i / steps)

    def forward(self, obs=None, y=None, steps=None):
        if steps is None:
            steps = self.steps

        state = self.init_state(obs, y)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, y, temp)
            
        out = self.get_output(state)
            
        return out, state
    
    def generate(self, y, steps=None):
        y = format_y(y, self.num_classes)
        _, state = self.forward(y=y, steps=steps)
        return state[0]['x']
    
    def classify(self, obs, steps=None):
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

# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierLI(FCClassifier):

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
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
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
        layers = []
        in_features = None
        for out_features in [self.input_size] + self.hidden_sizes:
            if self.precision_weighted:
                raise NotImplementedError
                layers.append(FCPW(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
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
        x = state[-1]['x']
        out = self.classifier(x.detach())
        return out


    def forward(self, obs=None, steps=None):
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp=temp)
            
        out = self.get_output(state)
            
        return out, state


    def classify(self, obs, steps=None):
        return self.forward(obs, steps)[0].argmax(dim=1)


    def reconstruct(self, obs, steps=None):
        if steps is None:
            steps = self.steps
        
        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, temp=temp)
        
        out = state[0]['x']

        return out, state

    
    def generate(self, y, steps=None):
        raise(NotImplementedError)


# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierSSLI(FCClassifierSS):

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
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

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
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
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()

    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
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
        return state[0]['x']

# Based on Whittington and Bogacz 2017
class FCClassifierInvSym(FCClassifierInv):
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
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
        
    # Initialises xs in state using 1 sweep of top-down predictions
    def _init_xs(self, state, obs=None, y=None):
        if obs is not None:
            for i, layer in reversed(list(enumerate(self.layers))):
                if i == len(self.layers) - 1: # last layer
                    state[i]['x'] = obs.clone()
                if i > 0:
                    state[i-1]['x'] = layer.predict_down(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = y.clone()
                else:
                    state[i]['x'] = layer.propagate_up(state[i-1]['x'])

# Based on Whittington and Bogacz 2017
class FCClassifierInvSS(FCClassifierInv):
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

    def init_layers(self):
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
        x = state[0]['x']
        out = self.classifier(x.detach())
        return out

    def forward(self, obs=None, steps=None):
        if steps is None:
            steps = self.steps

        state = self.init_state(obs)

        for i in range(steps):
            temp = self.calc_temp(i, steps)
            self.step(state, obs, temp=temp)
            
        out = self.get_output(state)
            
        return out, state
    
    def classify(self, obs, steps=None):
        return self.forward(obs, steps)[0].argmax(dim=1)
    
    def generate(self, y, steps=None):
        raise NotImplementedError("Generate not implemented for FCClassifierInvSS")
    