from typing import Any
from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import torch
import torch.nn as nn
from torch import Tensor, FloatTensor
from torch.autograd import Function
from torch.autograd.function import _ContextMethodMixin
import torch.nn.functional as F
import torch.optim as optim

total_num = 0

# ---------------------------------------------------------
# Helper Functions 
# ---------------------------------------------------------

class Differentiable_Round(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs):
        return grad_outputs

class Differentiable_Clip(Function):
    @staticmethod
    def forward(ctx: _ContextMethodMixin, input: Tensor, min_val: Tensor, max_val: Tensor) -> Any:
        ctx.save_for_backward(input, min_val, max_val)
        return torch.min(torch.max(input, min_val), max_val)

    @staticmethod
    def backward(ctx: _ContextMethodMixin, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensors
        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        
        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        
        if min_val.numel() == 1:
             grad_min = grad_min.sum().view(min_val.shape)
        else:
             target_shape = min_val.shape
             current_shape = grad_min.shape
             
             diff_dims = len(current_shape) - len(target_shape)
             if diff_dims > 0:
                 grad_min = grad_min.sum(dim=tuple(range(diff_dims)), keepdim=False)
             
             reduce_dims = [i for i, (c, t) in enumerate(zip(grad_min.shape, target_shape)) if t == 1 and c > 1]
             if reduce_dims:
                 grad_min = grad_min.sum(dim=tuple(reduce_dims), keepdim=True)

        if max_val.numel() == 1:
             grad_max = grad_max.sum().view(max_val.shape)
        else:
             target_shape = max_val.shape
             current_shape = grad_max.shape
             diff_dims = len(current_shape) - len(target_shape)
             if diff_dims > 0:
                 grad_max = grad_max.sum(dim=tuple(range(diff_dims)), keepdim=False)
             reduce_dims = [i for i, (c, t) in enumerate(zip(grad_max.shape, target_shape)) if t == 1 and c > 1]
             if reduce_dims:
                 grad_max = grad_max.sum(dim=tuple(reduce_dims), keepdim=True)
                 
        return grad_input, grad_min, grad_max

# ---------------------------------------------------------
# Functions for Bounds Initialization and Coarse Optimization
# ---------------------------------------------------------

def MinMaxInit(x: torch.Tensor, channel_wise: bool = False, is_weight: bool = False):
    if not channel_wise:
        return x.min(), x.max()
    if is_weight:
        reduce_dims = tuple(range(1, x.ndim))
    else:
        if x.ndim == 3: 
            reduce_dims = (0, 1)
        else:
            reduce_dims = tuple(i for i in range(x.ndim) if i != 1)
        
    lb = x.amin(dim=reduce_dims, keepdim=True)
    ub = x.amax(dim=reduce_dims, keepdim=True)
    return lb, ub

def MSE_Optimization_Search(input: torch.Tensor, bit: int, channel_wise: bool = False, is_weight: bool = False, num_steps: int = 100, lr: float = 0.01):

    with torch.no_grad():
        min_val, max_val = MinMaxInit(input, channel_wise, is_weight)
    
    lb = nn.Parameter(min_val.clone())
    ub = nn.Parameter(max_val.clone())
    
    with torch.enable_grad():
        
        optimizer = optim.Adam([lb, ub], lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.1)
        
        clip_fn = Differentiable_Clip.apply
        round_fn = Differentiable_Round.apply
        n_steps = (2 ** bit) - 1
        
        input = input.detach()

        for _ in range(num_steps):
            optimizer.zero_grad()
            
            range_val = ub - lb
            range_val = torch.max(range_val, torch.tensor(1e-6, device=input.device))
            s = range_val / n_steps
            
            c = clip_fn(input, lb, ub)
            q = round_fn((c - lb) / s)
            x_recon = q * s + lb
            
            loss = F.mse_loss(x_recon, input)
            
            loss.backward() 
            optimizer.step()
            scheduler.step()

    if not channel_wise:
        return float(lb.item()), float(ub.item())
    else:
        return lb.data, ub.data

# ---------------------------------------------------------
# Fake Quantizer Classes
# ---------------------------------------------------------

class FakeQuantizerBase(Module):
    def __init__(self, int_quant: bool = True, bit:int=4, shape=(1,), one_direction=False, metric='mse') -> None:
        super().__init__()
        self.lower_bound = Parameter(torch.randn(shape, dtype=torch.float32))
        self.upper_bound = Parameter(torch.randn(shape, dtype=torch.float32))
        self.n_bit = Parameter(torch.randn((1,), dtype=torch.float32))
        self.set_n_bit_manually(bit)
        
        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None
        self.metric = metric
        self.int_quant = int_quant
        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply
        self.calibrated = False
        self.one_direction_search = one_direction
        global total_num 
        total_num += 1
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for param_name in ['lower_bound', 'upper_bound']:
            key = prefix + param_name
            if key in state_dict:
                param = getattr(self, param_name)
                saved_param = state_dict[key]
                if saved_param.numel() == 1 and param.numel() > 1:
                     val = saved_param.item()
                     param.data.fill_(val)
                     state_dict[key] = torch.full_like(param.data, val)
                elif saved_param.shape != param.shape:
                     new_param = Parameter(torch.zeros_like(saved_param, device=param.device))
                     new_param.data.copy_(saved_param.data)
                     new_param.requires_grad = param.requires_grad
                     setattr(self, param_name, new_param)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.device
        if lb.shape != self.lower_bound.shape:
             if lb.numel() == 1: self.lower_bound.data.fill_(lb.item())
             else: self.lower_bound.data = lb.to(device)
        else:
            self.lower_bound.data = lb.clone().to(device)
        if ub.shape != self.upper_bound.shape:
             if ub.numel() == 1: self.upper_bound.data.fill_(ub.item())
             else: self.upper_bound.data = ub.to(device)
        else:
            self.upper_bound.data = ub.clone().to(device)

    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.device
        if isinstance(lb, float) or (isinstance(lb, Tensor) and lb.numel() == 1):
             val = lb if isinstance(lb, float) else lb.item()
             self.lower_bound.data.fill_(val)
        else:
             self.lower_bound.data = lb.to(device)
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.device
        if isinstance(ub, float) or (isinstance(ub, Tensor) and ub.numel() == 1):
             val = ub if isinstance(ub, float) else ub.item()
             self.upper_bound.data.fill_(val)
        else:
             self.upper_bound.data = ub.to(device)

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.device
        self.n_bit.data = FloatTensor([n_bit]).data.clone().to(device)


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self, bit=4, channel_wise=False, one_direction=False, metric='mse') -> None:
        super(FakeQuantizerWeight, self).__init__(bit=bit, shape=(1,), one_direction=one_direction, metric=metric)
        self.channel_wise = False 
        self.init_done = False
        self.is_weight = True

    def forward(self, x: torch.Tensor):
        if not self.calibrated:
            if self.init_done:
                return x
            
            # [Init] MSE-based Coarse Optimization
            best_lb, best_ub = MSE_Optimization_Search(
                x, 
                bit=int(self.n_bit.item()), 
                channel_wise=False,
                is_weight=True, 
                num_steps=100, 
                lr=0.01
            )
            
            if self.channel_wise:
                # Shape Mismatch
                if best_lb.numel() == 1 and self.lower_bound.numel() > 1:
                    # print(f"Warning: best_lb is scalar but lower_bound is vector. Expanding best_lb to match lower_bound shape {self.lower_bound.shape}.")
                    best_lb = best_lb.expand_as(self.lower_bound)
                    best_ub = best_ub.expand_as(self.upper_bound)
                
                if self.lower_bound.numel() == 1 and best_lb.numel() > 1:
                    # print(f"Warning: lower_bound is scalar but best_lb is vector. Expanding lower_bound to match best_lb shape {best_lb.shape}.")
                    self.lower_bound.data = torch.zeros_like(best_lb)
                    self.upper_bound.data = torch.zeros_like(best_ub)
                # print(f"Initialized Weight Quantizer Bounds (Channel-wise): LB shape {best_lb.shape}, UB shape {best_ub.shape}")
                self.lower_bound.data = best_lb
                self.upper_bound.data = best_ub
            else:
                # print(f"Initialized Weight Quantizer Bounds (Layer-wise): LB {best_lb}, UB {best_ub}")
                self.set_params_lb_manually(best_lb)
                self.set_params_ub_manually(best_ub)

            self.init_done = True
            return x
        
        # [Forward] Quantization
        if self.size_of_input is None:
            self.size_of_input = x.numel()
        
        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
        range_val = self.upper_bound - self.lower_bound
        range_val = torch.max(range_val, torch.tensor(1e-6, device=range_val.device))
        s = range_val / (torch.pow(2, n_bits) - 1)
        
        c = self.clip(x, self.lower_bound, self.upper_bound)
        r = self.round((c - self.lower_bound) / s)
        return s * r + self.lower_bound


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self, bit=4, channel_wise=False, one_direction=False, metric='mse') -> None:
        super(FakeQuantizerAct, self).__init__(bit=bit, shape=(1,), one_direction=one_direction, metric=metric)
        self.channel_wise = channel_wise
        self.running_stat = False 
        self.first_iter = False 
        self.dynamic = False
        self.beta = 0.995
        self.identity = False
        self.init_done = False 
        self.momentum = 0.1 
        self.is_weight = False 

    def forward(self, x):
        if self.identity:
            return x

        # [Init] MSE-based Coarse Optimization
        if not self.calibrated and not self.dynamic and not self.running_stat:
             best_lb, best_ub = MSE_Optimization_Search(
                 x, 
                 bit=int(self.n_bit.item()), 
                 channel_wise=self.channel_wise, 
                 is_weight=self.is_weight, 
                 num_steps=100, 
                 lr=0.01
             )
             
             if self.channel_wise:
                 if best_lb.numel() == 1 and self.lower_bound.numel() > 1:
                     best_lb = best_lb.expand_as(self.lower_bound)
                     best_ub = best_ub.expand_as(self.upper_bound)
                     
                 if self.lower_bound.numel() == 1 and best_lb.numel() > 1:
                     self.lower_bound.data = torch.zeros_like(best_lb)
                     self.upper_bound.data = torch.zeros_like(best_ub)
                 
                 if not self.init_done:
                     self.lower_bound.data = best_lb
                     self.upper_bound.data = best_ub
                     self.init_done = True
                 else:
                     # EMA Update
                     self.lower_bound.data = (1 - self.momentum) * self.lower_bound.data + self.momentum * best_lb
                     self.upper_bound.data = (1 - self.momentum) * self.upper_bound.data + self.momentum * best_ub
             else:
                 if not self.init_done:
                     self.set_params_lb_manually(best_lb)
                     self.set_params_ub_manually(best_ub)
                     self.init_done = True
                 else:
                     new_lb = torch.tensor(best_lb, device=self.lower_bound.device)
                     new_ub = torch.tensor(best_ub, device=self.upper_bound.device)
                     self.lower_bound.data = (1 - self.momentum) * self.lower_bound.data + self.momentum * new_lb
                     self.upper_bound.data = (1 - self.momentum) * self.upper_bound.data + self.momentum * new_ub
             
             return x
        
        if self.dynamic or (self.size_of_input is None and not self.calibrated):
             if self.calibrated: pass
             else:
                n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
                lb = torch.min(x).detach()
                ub = torch.max(x).detach()
                n_bits = n_bits.detach()
                range_val = ub - lb
                range_val = torch.max(range_val, torch.tensor(1e-6, device=x.device))
                s = range_val / (torch.pow(2, n_bits) - 1)
                c = self.clip(x, lb, ub)
                r = self.round((c - lb) / s)
                return s * r + lb

        if self.running_stat:
            with torch.no_grad():
                lb, ub = MinMaxInit(x, self.channel_wise, is_weight=False)
            
            if self.channel_wise:
                if self.lower_bound.numel() == 1 and lb.numel() > 1:
                     device = self.lower_bound.device
                     self.lower_bound.data = torch.zeros_like(lb, device=device)
                     self.upper_bound.data = torch.zeros_like(ub, device=device)
            
            if self.first_iter:
                self.lower_bound.data = lb.clone()
                self.upper_bound.data = ub.clone()
                self.first_iter = False
            else:
                self.lower_bound.data = self.beta * self.lower_bound.data + (1-self.beta) * lb
                self.upper_bound.data = self.beta * self.upper_bound.data + (1-self.beta) * ub
            return x

        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e: pass
            
        range_val = self.upper_bound - self.lower_bound
        range_val = torch.max(range_val, torch.tensor(1e-6, device=range_val.device))
        s = range_val / (torch.pow(2, n_bits) - 1)
        c = self.clip(x, self.lower_bound, self.upper_bound)
        r = self.round((c - self.lower_bound) / s)
        return s * r + self.lower_bound

# ---------------------------------------------------------
# Quantized Layer Wrappers
# ---------------------------------------------------------

class QuantBase(Module):
    def __init__(self,config):
        super().__init__()
        self.quant = True
        self.bit = config['bit']
        self.channel_wise = config.get('channel_wise', False) # Activation용
        self.channel_wise_weight = config.get('channel_wise_weight', False) # Weight용
        
        self.one_direction = config.get('one_direction', False)
        self.metric = config.get('metric', 'mse')
        self.metric_weight = config.get('metric_weight', self.metric)
        self.metric_act = config.get('metric_act', self.metric)
        
        self.weight_quantizer = FakeQuantizerWeight(self.bit, channel_wise=self.channel_wise_weight, one_direction=False, metric=self.metric_weight)
        self.act_quantizer = FakeQuantizerAct(self.bit, channel_wise=self.channel_wise, one_direction=self.one_direction, metric=self.metric_act)

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        self.weight_quantizer.set_require_grad(enable,enable, enable)
        self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias is not None:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        quant_weight = self.weight_quantizer(self.weight)
        return (quant_weight, self.bias)

class QuantLinear(QuantBase):
    def __init__(self,config):
        super().__init__(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        self.in_feature = linear.in_features
        self.out_feature = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data.clone())
        else:
            self.bias = linear.bias
            
        if self.act_quantizer.channel_wise:
             target_shape = (1, self.in_feature)
             if self.act_quantizer.lower_bound.numel() == 1 and self.in_feature > 1:
                  device = self.act_quantizer.lower_bound.device
                  self.act_quantizer.lower_bound = Parameter(torch.randn(target_shape, device=device))
                  self.act_quantizer.upper_bound = Parameter(torch.randn(target_shape, device=device))
                  self.act_quantizer.bound_shape = target_shape

        if self.weight_quantizer.channel_wise:
             target_shape = (self.out_feature, 1) 
             if self.weight_quantizer.lower_bound.numel() == 1 and self.out_feature > 1:
                  device = self.weight_quantizer.lower_bound.device
                  self.weight_quantizer.lower_bound = Parameter(torch.randn(target_shape, device=device))
                  self.weight_quantizer.upper_bound = Parameter(torch.randn(target_shape, device=device))

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        return F.linear(quant_act, quant_weight, self.bias)

class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2d):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.conv_kwargs = {
            "stride": conv.stride,
            "padding": conv.padding,
            "dilation": conv.dilation,
            "groups": conv.groups,
        }
        self.weight = Parameter(conv.weight.data.clone())
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data.clone())
        else:
            self.bias = conv.bias
        
        if self.act_quantizer.channel_wise:
             target_shape = (1, self.in_channels, 1, 1)
             if self.act_quantizer.lower_bound.numel() == 1 and self.in_channels > 1:
                  device = self.act_quantizer.lower_bound.device
                  self.act_quantizer.lower_bound = Parameter(torch.randn(target_shape, device=device))
                  self.act_quantizer.upper_bound = Parameter(torch.randn(target_shape, device=device))
        
        if self.weight_quantizer.channel_wise:
             target_shape = (self.out_channels, 1, 1, 1)
             if self.weight_quantizer.lower_bound.numel() == 1 and self.out_channels > 1:
                  device = self.weight_quantizer.lower_bound.device
                  self.weight_quantizer.lower_bound = Parameter(torch.randn(target_shape, device=device))
                  self.weight_quantizer.upper_bound = Parameter(torch.randn(target_shape, device=device))

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        return F.conv2d(quant_act, quant_weight, self.bias, **self.conv_kwargs)

class QuantLinearQKV(Module):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    

    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()
        
        quant_weight = torch.cat([w_q, w_k, w_v],dim=0)
        if b_q is not None:
            bias = torch.cat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):

        self.in_feature = linear.in_features 
        self.out_feature = linear.out_features // 3
        
        linear_q = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_k = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        linear_v = Linear(self.in_feature, self.out_feature,bias=linear.bias is not None)
        
        
        linear_q.weight.data,linear_k.weight.data,linear_v.weight.data = linear.weight.data.clone().reshape(3,self.in_feature,self.out_feature)
        
        
        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)
    
        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        return torch.cat([q,k,v], dim=-1)
    
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)

        return F.linear(quant_act, quant_weight, self.bias)

class QuantConv2dQKV(Module):
    def __init__(self, config):
        super().__init__()
        self.q = QuantConv2d(config)
        self.k = QuantConv2d(config)
        self.v = QuantConv2d(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)

    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    
    def get_quant_weight_bias(self):
        w_q, b_q = self.q.get_quant_weight_bias()
        w_k, b_k = self.k.get_quant_weight_bias()
        w_v, b_v = self.v.get_quant_weight_bias()
        
        quant_weight = torch.cat([w_q, w_k, w_v], dim=0)
        bias = torch.cat([b_q, b_k, b_v]) if b_q is not None else None
        return (quant_weight, bias)

    def set_param(self, conv: Conv2d):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels // 3
        
        conv_q = Conv2d(self.in_channels, self.out_channels, 1, bias=conv.bias is not None)
        conv_k = Conv2d(self.in_channels, self.out_channels, 1, bias=conv.bias is not None)
        conv_v = Conv2d(self.in_channels, self.out_channels, 1, bias=conv.bias is not None)
        
        conv_q.weight.data, conv_k.weight.data, conv_v.weight.data = conv.weight.data.chunk(3, dim=0)
        if conv.bias is not None:
            conv_q.bias.data, conv_k.bias.data, conv_v.bias.data = conv.bias.data.chunk(3, dim=0)
    
        self.q.set_param(conv_q)
        self.k.set_param(conv_k)
        self.v.set_param(conv_v)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.cat([q, k, v], dim=1)
from einops import rearrange

class QuantWindowAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = None
        self.window_size = None
        self.num_heads = None
        self.scale = None
        self.quant = True
        
        self.qkv = None 
        self.proj = None
        self.softmax = None
        self.attn_drop = None
        self.proj_drop = None
        
        # Q, K, V Activation Quantizers
        self.q_quantizer = FakeQuantizerAct(config['bit'], channel_wise=config.get('channel_wise', False), metric=config.get('metric_act', 'mse'), one_direction=config.get('one_direction', False))
        self.k_quantizer = FakeQuantizerAct(config['bit'], channel_wise=config.get('channel_wise', False), metric=config.get('metric_act', 'mse'), one_direction=config.get('one_direction', False))
        self.v_quantizer = FakeQuantizerAct(config['bit'], channel_wise=config.get('channel_wise', False), metric=config.get('metric_act', 'mse'), one_direction=config.get('one_direction', False))
        self.attn_quantizer = FakeQuantizerAct(config['bit'], channel_wise=config.get('channel_wise', False), metric=config.get('metric_act', 'mse'), one_direction=config.get('one_direction', False))

    def set_param(self, attn_module):
        self.dim = attn_module.dim
        self.window_size = attn_module.window_size
        self.num_heads = attn_module.num_heads
        self.scale = attn_module.scale
        
        # Copy buffers (relative_position_index, attn_mask if exists)
        self.register_buffer("relative_position_index", attn_module.relative_position_index)
        if hasattr(attn_module, "attn_mask"):
            self.register_buffer("attn_mask", attn_module.attn_mask)
            
        # Parameter Table for Relative Position Bias
        self.relative_position_bias_table = Parameter(attn_module.relative_position_bias_table.data.clone())

        # Quantized Layers
        self.qkv = QuantLinear(self.config)
        self.qkv.set_param(attn_module.qkv)
        
        self.proj = QuantLinear(self.config)
        self.proj.set_param(attn_module.proj)
        
        self.softmax = attn_module.softmax
        self.attn_drop = attn_module.attn_drop
        self.proj_drop = attn_module.proj_drop

    def set_quant_flag(self, enable: bool):
        self.quant = enable
        if self.qkv: self.qkv.set_quant_flag(enable)
        if self.proj: self.proj.set_quant_flag(enable)

    def set_require_grad(self, enable: bool):
        self.q_quantizer.set_require_grad(enable, enable, enable)
        self.k_quantizer.set_require_grad(enable, enable, enable)
        self.v_quantizer.set_require_grad(enable, enable, enable)
        self.attn_quantizer.set_require_grad(enable, enable, enable)
        if self.qkv: self.qkv.set_require_grad(enable)
        if self.proj: self.proj.set_require_grad(enable)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # QuantLinear for qkv
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        
        # Quantize Q, K, V
        if self.quant:
            q = self.q_quantizer(q)
            k = self.k_quantizer(k)
            v = self.v_quantizer(v)
            
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # Quantize Attention Map (Optional but usually good for full quant)
        if self.quant:
            attn = self.attn_quantizer(attn)
            
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # QuantLinear for proj
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
