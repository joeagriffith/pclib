import torch
import torch.nn.functional as F
import numpy as np

def reTanh(x:torch.Tensor):
    """
    | Applies the tanh then relu function element-wise:
    | x = x.tanh().relu()

    Parameters
    ----------
        x : torch.Tensor
    
    Returns
    -------
        torch.Tensor
    """
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

class Shrinkage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.save_for_backward(input, lambda_)
        signs = input.sign()
        # set vals < or > lambda_ to 0
        output = input.abs() - lambda_
        output = output.clamp(min=0)
        return output * signs

    @staticmethod
    def backward(ctx, grad_output):
        input, lambda_ = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() < lambda_] = 0
        return grad_input, grad_output

# To apply this function
def shrinkage(input, lambda_=1.0):
    return Shrinkage.apply(input, lambda_)

def d_shrinkage(input, lambda_=1.0):
    return (input.abs() > lambda_).float()

# Calculate Correlations
def calc_corr(state):
    correlations = []
    for state_i in state:
        # Standardise activations
        mean = state_i['x'].mean(dim=0, keepdim=True)
        std = state_i['x'].std(dim=0, keepdim=True) + 1e-5
        x = (state_i['x'] - mean) / std

        # Compute Correlation matrix
        corr_matrix = torch.corrcoef(x.T)
        mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
        correlations.append(torch.nanmean(corr_matrix.masked_select(mask).abs()))
    
    return sum(correlations) / len(correlations)

def calc_sparsity(state):
    num_zeros = [(state_i['x'].numel() - torch.count_nonzero(state_i['x'])) / state_i['x'].numel() for state_i in state]
    return sum(num_zeros) / len(num_zeros)
