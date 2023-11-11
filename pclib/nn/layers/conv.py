import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import Optional

# Whittington & Bogacz 2017
class Conv2d(nn.Module):
    __constants__ = ['shape', 'prev_shape']
    shape: int
    prev_shape: Optional[int]

    def __init__(self,
                 shape: (int, int, int), # (channels, height, width)
                 prev_channels: int = None,
                 kernel_size: int = 3,
                 padding = 'same',
                 stride: int = 1,
                 bias: bool = True,
                 maxpool=1,
                 actv_fn: callable = F.tanh,
                 d_actv_fn: callable = None,
                 gamma: float = 0.1,
                 beta: float = 1.0,
                 device=torch.device('cpu'),
                 dtype=None
                 ) -> None:
        
        assert stride == 1, "Stride > 1 not yet supported."

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d, self).__init__()

        self.shape = shape
        # self.prev_shape = prev_shape
        self.actv_fn = actv_fn
        self.gamma = gamma
        self.beta = beta
        self.device = device


        # Default derivative of activation function
        if d_actv_fn is not None:
            self.d_actv_fn: callable = d_actv_fn
        elif actv_fn == F.relu:
            self.d_actv_fn: callable = lambda x: torch.sign(torch.relu(x))
        elif actv_fn == F.sigmoid:
            self.d_actv_fn: callable = lambda x: torch.sigmoid(x) * (1 - torch.sigmoid(x))
        elif actv_fn == F.tanh:
            self.d_actv_fn: callable = lambda x: 1 - torch.tanh(x).square()
        
        # Initialise weights if not input layer
        if prev_channels is not None:
            self.conv_td = nn.Sequential(
                nn.Conv2d(shape[0], prev_channels, kernel_size, padding=padding, stride=stride, bias=bias, **factory_kwargs),
                nn.MaxPool2d(kernel_size=maxpool),
            )
            # self.conv_bu = nn.Sequential(
            #     nn.Upsample(scale_factor=maxpool),
            #     nn.ConvTranspose2d(prev_shape[0], shape[0], kernel_size, padding=padding, stride=stride, bias=False, **factory_kwargs),
            # )

    def init_state(self, batch_size):
        return {
            'x': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device, requires_grad=True),
            'e': torch.zeros((batch_size, self.shape[0], self.shape[1], self.shape[2]), device=self.device),
        }

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    # Returns a prediction of the state in the previous layer
    def predict(self, state):
        return self.conv_td(self.actv_fn(state['x']))
    
    # propagates error from layer below, return an update for x
    def propagate(self, e_below):
        # return self.conv_bu(e_below)
        raise(NotImplementedError("propagate not implemented for Conv2d"))
    
    # Recalculates prediction-error between state and top-down prediction of it
    # With simulated annealing
    def update_e(self, state, pred=None, temp=None):
        if pred is not None:
            state['e'] = state['x'] - pred

        if temp is not None:
            eps = torch.randn_like(state['e'], device=self.device) * 0.034 * temp
            state['e'] += eps

    def update_x(self, state, e_below=None):
        state['x'] = state['x'].detach() - self.gamma * state['x'].grad
        state['x'].requires_grad = True

    # def update_x(self, state, e_below=None):
        # # If not input layer, propagate error from layer below
        # if e_below is not None:
        #     if e_below.dim() == 2:
        #         e_below = e_below.unsqueeze(-1).unsqueeze(-1)
        #     update = self.propagate(e_below)
        #     state['x'] += self.gamma * (-state['e'] + update * self.d_actv_fn(state['x']))
        # # This update will be zero if top layer
        # state['x'] += self.gamma * (self.beta * -state['e'])
        
    # def update_grad(self, state, e_below):
    #     if state['e'].norm() > 0:
    #         state['e'].square().sum().backward()



















# ## A few different implementations of a convolutional layer, ALL DEPRECATED BUT TEMPORARILY KEPT FOR REFERENCE
# class Conv2dV1(nn.Module):
#     def __init__(self, 
#                  in_shape,
#                  out_shape,
#                  kernel,
                 
#                  nu,
#                  mu,
#                  eta,

#                  maxpool=1,
#                  bu_actv=nn.Tanh(),
#                  td_actv=nn.Tanh(),
                 
#                  padding=0,
#                  relu_errs=True,
#                  **kwargs,
#                 ):
#         super(Conv2dV1, self).__init__()
#         self.r_shape = out_shape
#         self.e_shape = in_shape

#         self.nu = nu
#         self.mu = mu
#         self.eta = eta

#         self.relu_errs = relu_errs
#         self.device = "cpu"

#         self.bottomUp = nn.Sequential(
#             nn.Conv2d(in_shape[0], out_shape[0], kernel, padding=padding, **kwargs),
#             nn.MaxPool2d(kernel_size=maxpool),
#             bu_actv,
#         )
        
#         self.topDown = nn.Sequential(
#             nn.Upsample(scale_factor=maxpool),
#             nn.ConvTranspose2d(out_shape[0], in_shape[0], kernel, padding=padding, **kwargs),
#             td_actv,
#         )
    
#     def init_vars(self, batch_size):
#         r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2])).to(self.device)
#         e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2])).to(self.device)
#         return r, e

#     def to(self, *args, **kwargs):
#         self.device = args[0]
#         return super().to(*args, **kwargs)
    
#     def forward(self, x, r, td_err=None):
#         e = x - self.topDown(r)
#         if self.relu_errs:
#             e = F.relu(e)
#         r = self.nu*r + self.mu*self.bottomUp(e)
#         if td_err is not None:
#             if td_err.dim() == 2 and r.dim() == 4:
#                 td_err = td_err.unsqueeze(-1).unsqueeze(-1)
#             r += self.eta*td_err
#         return r, e

# class Conv2dV2(nn.modules.conv._ConvNd):

#     def __init__(
#             self, 
#             e_shape,
#             r_shape,
#             kernel_size: _size_2_t,
#             stride: _size_2_t = 1,
#             padding: Union[str, _size_2_t] = 0,
#             dilation: _size_2_t = 1,
#             groups: int = 1,
#             bias: bool = True,
#             padding_mode: str = 'zeros',
#             device=None,
#             dtype=None,
                 
#             nu=1.0,
#             mu=0.2,
#             eta=0.05,

#             td_actv=nn.Tanh(),

#             maxpool=None,
#             relu_errs=True,
#         ) -> None:
#         self.factory_kwargs = {'device': device, 'dtype': dtype}
#         kernel_size_ = _pair(kernel_size)
#         stride_ = _pair(stride)
#         padding_ = padding if isinstance(padding, str) else _pair(padding)
#         dilation_ = _pair(dilation)
#         super(Conv2dV2, self).__init__(
#             e_shape[0], r_shape[0], kernel_size_, stride_, padding_, dilation_,
#             False, _pair(0), groups, bias, padding_mode, **self.factory_kwargs
#         )
#         self.e_shape = e_shape
#         self.r_shape = r_shape

#         self.downsample = nn.MaxPool2d(kernel_size=maxpool)
#         self.upsample = nn.Upsample(scale_factor=maxpool)

#         self.nu = nu
#         self.mu = mu
#         self.eta = eta
#         self.relu_errs = relu_errs

#         self.td_actv = td_actv
#         self.device = device
    
#     def init_vars(self, batch_size):
#         r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2]), **self.factory_kwargs)
#         e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2]), **self.factory_kwargs)
#         return r, e

#     def _conv_forward(self, input: Tensor):
#         if self.padding_mode != 'zeros':
#             return self.downsample(F.conv2d(
#                 F.pad(input, self.reversed_padding_repeated_twice, mode=self.padding_mode),
#                 self.weight,
#                 None,
#                 self.stride,
#                 self.padding,
#                 self.dilation,
#                 self.groups,                            
#             ))
#         else:
#             return self.downsample(F.conv2d(
#                 input,
#                 self.weight,
#                 None,
#                 self.stride,
#                 self.padding,
#                 self.dilation,
#                 self.groups,
#             ))

#     def _output_padding(self, input: Tensor, output_size: Optional[List[int]],
#                             stride: List[int], padding: List[int], kernel_size: List[int],
#                             num_spatial_dims: int, dilation: Optional[List[int]] = None) -> List[int]:
#             if output_size is None:
#                 ret = _single(self.output_padding)  # converting to list if was not already
#             else:
#                 has_batch_dim = input.dim() == num_spatial_dims + 2
#                 num_non_spatial_dims = 2 if has_batch_dim else 1
#                 if len(output_size) == num_non_spatial_dims + num_spatial_dims:
#                     output_size = output_size[num_non_spatial_dims:]
#                 if len(output_size) != num_spatial_dims:
#                     raise ValueError(
#                         "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})"
#                         .format(num_spatial_dims, input.dim(), num_spatial_dims,
#                                 num_non_spatial_dims + num_spatial_dims, len(output_size)))

#                 min_sizes = torch.jit.annotate(List[int], [])
#                 max_sizes = torch.jit.annotate(List[int], [])
#                 for d in range(num_spatial_dims):
#                     dim_size = ((input.size(d + num_non_spatial_dims) - 1) * stride[d] -
#                                 2 * padding[d] +
#                                 (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
#                     min_sizes.append(dim_size)
#                     max_sizes.append(min_sizes[d] + stride[d] - 1)

#                 for i in range(len(output_size)):
#                     size = output_size[i]
#                     min_size = min_sizes[i]
#                     max_size = max_sizes[i]
#                     if size < min_size or size > max_size:
#                         raise ValueError((
#                             "requested an output size of {}, but valid sizes range "
#                             "from {} to {} (for an input of {})").format(
#                                 output_size, min_sizes, max_sizes, input.size()[2:]))

#                 res = torch.jit.annotate(List[int], [])
#                 for d in range(num_spatial_dims):
#                     res.append(output_size[d] - min_sizes[d])

#             return res

#     def _conv_backward(self, input: Tensor):
#         if self.padding_mode != 'zeros':
#             raise ValueError('Only "zeros" padding mode is supported for ConvTranspose2d')
#         assert isinstance(self.padding, tuple)
#         num_spatial_dims = 2
#         output_padding = self._output_padding(
#             input, self.e_shape, self.stride, self.padding, 
#             self.kernel_size, num_spatial_dims, self.dilation
#         )  
#         return self.upsample(F.conv_transpose2d(
#             input, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
#             output_padding=output_padding, groups=self.groups, dilation=self.dilation
#         ))
    
#     def forward(self, x, r, td_err=None, r_no_grad=False):
#         pred = self.td_actv(self._conv_backward(r))
#         e = x - pred
#         if self.relu_errs:
#             e = F.relu(e)
#         r = self.nu*r + self.mu*self._conv_forward(e)
#         if td_err is not None:
#             r += self.eta*td_err
#         return r, e

# class Conv2dV3(nn.Module):
#     def __init__(self, 
#                  e_shape,
#                  r_shape,
#                  kernel,
                 
#                  nu,
#                  mu,
#                  eta,

#                  maxpool=1,
#                  forw_actv=nn.ReLU(),
#                  td_actv=nn.Tanh(),
                 
#                  padding=0,
#                  **kwargs,
#                 ):
#         super(Conv2dV3, self).__init__()
#         self.e_shape = e_shape
#         self.r_shape = r_shape

#         self.nu = nu
#         self.mu = mu
#         self.eta = eta

#         self.device = "cpu"

#         self.conv = nn.Sequential(
#             nn.Conv2d(e_shape[0], r_shape[0], kernel, padding=padding, **kwargs),
#             nn.MaxPool2d(kernel_size=maxpool)
#         )
#         self.forw_actv = forw_actv
        
#         self.convT = nn.Sequential(
#             nn.Upsample(scale_factor=maxpool),
#             nn.ConvTranspose2d(r_shape[0], e_shape[0], kernel, padding=padding, **kwargs),
#             td_actv,
#         )

#         self.rec_conv = nn.Sequential(
#             nn.Conv2d(r_shape[0], r_shape[0], (10,10), padding="same")
#         )
    
#     # def to(self, device):
#     #     self.conv = self.conv.to(device)
#     #     self.convT = self.convT.to(device)
#     #     self.rec_conv = self.rec_conv.to(device)
#     #     self.device = device
    
#     def init_vars(self, batch_size):
#         r = torch.zeros((batch_size, self.r_shape[0], self.r_shape[1], self.r_shape[2])).to(self.device)
#         e = torch.zeros((batch_size, self.e_shape[0], self.e_shape[1], self.e_shape[2])).to(self.device)
#         return r, e
    
#     def forward(self, x, r, td_err=None):
#         e = self.forw_actv(x - self.convT(r))
#         # r = self.nu*self.rec_conv(r) + self.mu*self.conv(e)
#         r = self.nu*r + self.mu*self.conv(e)
#         if td_err is not None:
#             r -= self.eta*td_err
#         return r, e