from pclib.nn.layers import PrecisionWeighted as PrecisionWeighted
from pclib.nn.layers import LinearSym
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017
class PCLinearClassifierSym(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    num_classes: int

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super(PCLinearClassifierSym, self).__init__()

        self.in_features = input_size
        self.num_classes = num_classes
        self.bias = bias
        self.symmetric = symmetric
        self.precision_weighted = precision_weighted
        self.gamma = gamma
        self.beta = beta

        layers = []
        sizes = [num_classes] + hidden_sizes + [input_size]
        for i, size in enumerate(sizes):
            prev_size = sizes[i-1] if i > 0 else None
            next_size = sizes[i+1] if i < len(sizes) - 1 else None
            if precision_weighted:
                layers.append(PrecisionWeighted(size, prev_size, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(LinearSym(size, prev_size, next_size, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))

        self.layers = nn.ModuleList(layers)
        self.steps = steps
        self.device = device

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
        
        # Pin input and output Xs if provided
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
                    state[i-1]['x'] = layer.predict_down(state[i])
            if y is not None:
                state[0]['x'] = y.clone()
        elif y is not None:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    state[0]['x'] = y.clone()
                elif i < len(self.layers) - 1:
                    state[i]['x'] = layer.propagate_up(state[i-1]['x'])

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
        return state[0]['x']

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
    
    def classify(self, obs, state=None, steps=None):
        assert len(obs.shape) == 2, f"Input must be 2D, got {len(obs.shape)}D"

        if steps is None:
            steps = self.steps

        vfes = torch.zeros(obs.shape[0], self.num_classes, device=self.device)
        for target in range(self.num_classes):
            targets = torch.full((obs.shape[0],), target, device=self.device, dtype=torch.long)
            y = format_y(targets, self.num_classes)
            _, state = self.forward(obs, y, steps)
            vfes[:, target] = vfe(state, batch_reduction=None)
        
        return vfes.argmin(dim=1)

