import torch
import torch.nn.functional as F

def reTanh(x):
    return x.tanh().relu()

def identity(x):
    return x

def trec(x):
    return x * (x > 1.0).float()

# Output e.g. [0.03, 0.03, 0.97] for num_classes=3 and target=2
def format_y(targets, num_classes):
    assert len(targets.shape) == 1, f"Targets must be 1D, got {len(targets.shape)}D"
    targets = F.one_hot(targets, num_classes).float()
    baseline = torch.ones_like(targets) * 0.03
    y = baseline + (targets * 0.94)
    return y


class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Modify here to allow gradients to flow freely.
        # For example, you might want to pass all gradients through:
        grad_input[input < 0] = grad_output[input < 0]
        return grad_input

# To apply this function
def my_relu(input):
    return CustomReLU.apply(input)


# Calculate Correlations
def calc_corr(state):
    # Normalize activations
    activations = [F.normalize(state_i['x'].flatten(1), dim=1) for state_i in state]
    # Calculate correlations
    correlations = [torch.corrcoef(activations_i.t()) for activations_i in activations]
    # Mask to exclude self-correlations
    masks = [torch.eye(corr.shape[0]).to(correlations[0].device) for corr in correlations]
    masked_correlations = [corr.masked_select(mask == 0).abs().mean() for corr, mask in zip(correlations, masks)]
    # Return average absolute correlation
    return sum(masked_correlations) / len(masked_correlations)


def calc_sparsity(state, std_multiplier=0.1):
    thresholds = [std_multiplier * state_i['x'].std() for state_i in state]
    small_values = [(state_i['x'].abs() < threshold).sum(dim=1).float().mean() for state_i, threshold in zip(state, thresholds)]
    return sum(small_values) / len(small_values)




