from pclib.nn.layers import FCSym, FCPW
from pclib.nn.models import FCClassifier
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017
class FCClassifierInvSym(FCClassifier):
    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'has_bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

        layers = []
        features = [num_classes] + hidden_sizes + [input_size]
        for i, out_features in enumerate(features):
            in_features = features[i-1] if i > 0 else None
            next_features = features[i+1] if i < len(features) - 1 else None
            if precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCSym layers")
            else:
                layers.append(FCSym(in_features, out_features, next_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
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
        
    def pin(self, state, obs=None, y=None):
        if obs is not None:
            state[-1]['x'] = obs.clone()
        if y is not None:
            state[0]['x'] = y.clone()

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

    def get_output(self, state):
        return state[0]['x']