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
def shrinkage(input, lambda_=1e-3):
    return Shrinkage.apply(input, lambda_)

def d_shrinkage(input, lambda_=1e-3):
    return (input.abs() > lambda_).float()

# Calculate Correlation between activations of top latent vector
def calc_corr(state):
    # Standardise activations
    mean = state[-1]['x'].mean(dim=0, keepdim=True)
    std = state[-1]['x'].std(dim=0, keepdim=True) + 1e-5
    x = (state[-1]['x'] - mean) / std

    # Compute Correlation matrix
    corr_matrix = torch.corrcoef(x.T)
    mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
    correlation = torch.nanmean(corr_matrix.masked_select(mask).abs())
    
    return correlation

def calc_sparsity(state):
    num_zeros = [(state_i['x'].numel() - torch.count_nonzero(state_i['x'])) / state_i['x'].numel() for state_i in state[1:]]
    return sum(num_zeros) / len(num_zeros)

# Will mix two numbers of same class 1/num_classes of the time
class MakeNegativeSample():
    def __init__(self, dataset):
        self.dataset = dataset
        self.n = len(dataset)
    def __call__(self, x, steps=10):
        i = torch.randint(0, self.n-1, (x.shape[0],))
        return mix_images(x, self.dataset[i][0], steps=steps)

def mix_images(x1, x2, steps=10, return_mask=False):
    device = x1.device
    mask = torch.bernoulli(torch.ones((x1.shape[0],1,28,28))*0.5).to(device)
    # blur  with a filter of the form [1/4, 1/2, 1/4] in both horizontal and veritical directions
    filter_h = torch.tensor([[1/4, 1/2, 1/4]]).unsqueeze(0).to(device)
    filter_v = torch.tensor([[1/4], [1/2], [1/4]]).unsqueeze(0).to(device)
    for _ in range(steps):
        mask = F.conv2d(mask, filter_h.unsqueeze(0), padding='same')
        mask = F.conv2d(mask, filter_v.unsqueeze(0), padding='same')
    
    # threshold at 0.5
    mask_1 = mask > 0.5
    mask_2 = mask <= 0.5

    out = x1*mask_1 + x2*mask_2
    if return_mask:
        return out.squeeze(0), mask_1
    else:
        return out.squeeze(0)