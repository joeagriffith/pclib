from pclib.nn.layers import FC, FCPW
from pclib.nn.models import FCClassifier
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017
class FCClassifierInv(FCClassifier):

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'has_bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

        layers = []
        in_features = None
        for out_features in [num_classes] + hidden_sizes + [input_size]:
            if precision_weighted:
                layers.append(FCPW(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            else:
                layers.append(FC(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
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