"""
Wavelet subband normalization module
Adapted from wavelet-ldct-denoising-main/models/convs/norm.py

Per-subband normalization for fair loss computation across frequency bands
"""

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class _Norm(nn.Module):
    def __init__(
        self,
        num_features,
        eps=1e-3,
        momentum=0.01,
        affine=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        self.register_buffer(
            'running_mean', torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            'running_var', torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            'num_batches_tracked', torch.tensor(0, dtype=torch.long)
        )
        
        if self.affine:
            self.weight = nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )
            self.bias = nn.Parameter(
                torch.zeros(num_features, dtype=torch.float)
            )
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def _check_input_dim(self, input):
        raise NotImplementedError()

    def forward(self, x, update_stat=False, inverse=False):
        """
        Forward normalization with optional inverse
        
        Args:
            x: Input tensor
            update_stat: Whether to update running statistics (during training)
            inverse: Whether to denormalize (False: normalize, True: denormalize)
        
        Returns:
            Normalized or denormalized tensor
        """
        self._check_input_dim(x)
        
        # Transpose to channel-last for normalization
        if x.dim() > 2:
            x = x.transpose(1, -1)
        
        if update_stat:
            # Update running statistics
            dims = [i for i in range(x.dim() - 1)]
            
            with torch.no_grad():
                batch_mean = x.mean(dims)
                batch_var = x.var(dims, unbiased=False)
                
                if self.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = self.momentum

                self.running_mean = (
                    exponential_average_factor * batch_mean +
                    (1 - exponential_average_factor) * self.running_mean
                )
                self.running_var = (
                    exponential_average_factor * batch_var +
                    (1 - exponential_average_factor) * self.running_var
                )
                self.num_batches_tracked += 1
        
        # Normalize or denormalize
        if not inverse:
            # Normalize: (x - mean) / sqrt(var)
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        else:
            # Denormalize: x * sqrt(var) + mean
            x = x * torch.sqrt(self.running_var + self.eps) + self.running_mean
        
        # Transpose back
        if x.dim() > 2:
            x = x.transpose(1, -1)
        
        return x


class Norm1d_(_Norm):
    """1D normalization for 2D or 3D input (N, C) or (N, C, L)"""
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class Norm2d_(_Norm):
    """2D normalization for 4D input (N, C, H, W)"""
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class Norm3d_(_Norm):
    """3D normalization for 5D input (N, C, D, H, W)"""
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))



# class _Norm(torch.jit.ScriptModule):
#     def __init__(
#         self,
#         num_features: int,
#         eps: float = 1e-3,
#         momentum: float = 0.01,
#         affine: bool = True,
#     ):
#         super().__init__()
#         self.register_buffer(
#             "running_mean", torch.zeros(num_features, dtype=torch.float)
#         )
#         self.register_buffer(
#             "running_std", torch.ones(num_features, dtype=torch.float)
#         )
#         self.register_buffer(
#             "num_batches_tracked", torch.tensor(0, dtype=torch.long)
#         )
#         self.weight = torch.nn.Parameter(
#             torch.ones(num_features, dtype=torch.float)
#         )
#         self.bias = torch.nn.Parameter(
#             torch.zeros(num_features, dtype=torch.float)
#         )
#         self.eps = eps
#         self.step = 0
#         self.momentum = momentum

#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         raise NotImplementedError()  # pragma: no cover

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         self._check_input_dim(x)
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         if self.training:
#             dims = [i for i in range(x.dim() - 1)]
#             batch_mean = x.mean(dims)
#             batch_std = x.std(dims, unbiased=False) + self.eps
            
#             self.running_mean += self.momentum * (
#                 batch_mean.detach() - self.running_mean
#             )
#             self.running_std += self.momentum * (
#                 batch_std.detach() - self.running_std
#             )
#             self.num_batches_tracked += 1
#         x = (x - self.running_mean) / self.running_std
        
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         return x

#     def inverse(self, x: torch.Tensor) -> torch.Tensor:
#         self._check_input_dim(x)
#         if x.dim() > 2:
#             x = x.transpose(1, -1)

#         x = x * self.running_std + self.running_mean
        
#         if x.dim() > 2:
#             x = x.transpose(1, -1)
#         return x

# class Norm1d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() not in [2, 3]:
#             raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


# class Norm2d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() != 4:
#             raise ValueError("expected 4D input (got {x.dim()}D input)")


# class Norm3d(_Norm):
#     def _check_input_dim(self, x: torch.Tensor) -> None:
#         if x.dim() != 5:
#             raise ValueError("expected 5D input (got {x.dim()}D input)")