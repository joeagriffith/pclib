import torch
import torch.nn.functional as F

def vfe(state, batch_reduction='mean', layer_reduction='sum'):
    if layer_reduction == 'sum':
        vfe = sum([state_i['e'].square().sum(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
    elif layer_reduction =='mean':
        vfe = sum([state_i['e'].square().mean(dim=[i for i in range(1, state_i['e'].dim())]) for state_i in state])
    if batch_reduction == 'sum':
        vfe = vfe.sum()
    elif batch_reduction == 'mean':
        vfe = vfe.mean()

    return vfe

# Output e.g. [0.03, 0.03, 0.97] for num_classes=3 and target=2
def format_y(targets, num_classes):
    assert len(targets.shape) == 1, f"Targets must be 1D, got {len(targets.shape)}D"
    targets = F.one_hot(targets, num_classes).float()
    baseline = torch.ones_like(targets) * 0.03
    y = baseline + (targets * 0.94)
    return y