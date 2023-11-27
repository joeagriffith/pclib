from pclib.nn.layers import FCLI, FCPW
from pclib.nn.models import FCClassifier
from pclib.utils.functional import vfe, format_y
import torch
import torch.nn as nn
import torch.nn.functional as F


# Based on Whittington and Bogacz 2017, but with targets predicting inputs
class FCClassifierLI(FCClassifier):

    def __init__(self, input_size, num_classes, hidden_sizes = [], steps=20, bias=True, symmetric=True, precision_weighted=False, actv_fn=F.relu, d_actv_fn=None, gamma=0.1, beta=1.0, device=torch.device('cpu'), dtype=None):
        factory_kwargs = {'has_bias': bias, 'symmetric': symmetric, 'device': device, 'dtype': dtype}
        super().__init__(input_size, num_classes, hidden_sizes, steps, bias, symmetric, precision_weighted, actv_fn, d_actv_fn, gamma, beta, device, dtype)

        layers = []
        in_features = None
        for out_features in [input_size] + hidden_sizes + [num_classes]:
            if precision_weighted:
                raise NotImplementedError("Precision weighted not implemented for FCClassifierLI")
            else:
                layers.append(FCLI(in_features, out_features, actv_fn=actv_fn, d_actv_fn=d_actv_fn, gamma=gamma, beta=beta, **factory_kwargs))
            in_features = out_features
        self.layers = nn.ModuleList(layers)