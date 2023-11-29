import torch
import torch.nn.functional as F

def reTanh(x):
    return x.tanh().relu()

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
    corrs = [F.relu(state_i['x']).t().corrcoef() for state_i in state]
    corrs = [corr.nan_to_num() for corr in corrs]
    avg_corr = sum([(corr - torch.eye(corr.shape[0]).to(corr.device)).abs().sum() for corr in corrs]) / len(corrs)
    return avg_corr


