import torch.nn as nn
from .quant_arch import QuantConv2d, QuantLinear, FakeQuantizerBase, QuantLinearQKV, QuantConv2dQKV, QuantWindowAttention
from models.network_swinir import WindowAttention


def convert_to_quantized(module, bit, layer_name=""):
    # ---------------------------------------------------------
    # SwinIR은 모든 activation channel-wise quantization
    # ---------------------------------------------------------
    is_cw_act = True 

    config = {
        'bit': bit, 
        'channel_wise': is_cw_act,    
        'channel_wise_weight': False, # Weight은 모두 Layer-wise 고정
        'metric': 'mse'
    }
    # ---------------------------------------------------------
    
    if isinstance(module, nn.Conv2d):
        q_layer = QuantConv2d(config)
        q_layer.set_param(module)
        return q_layer
    elif isinstance(module, nn.Linear):
        q_layer = QuantLinear(config)
        q_layer.set_param(module)
        return q_layer
        
    for name, submodule in module.named_children():
        sub_layer_name = layer_name + "." + name if layer_name != "" else name
        
        if isinstance(submodule, WindowAttention):
            q_attn = QuantWindowAttention(config)
            q_attn.set_param(submodule)
            setattr(module, name, q_attn)
            continue

        
        setattr(module, name, convert_to_quantized(submodule, bit, sub_layer_name))
            
    return module

def enable_calibration(module):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, FakeQuantizerBase):
            sub_module.calibrated = False

def get_quant_params(module):
    params = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, FakeQuantizerBase):
            if sub_module.lower_bound.requires_grad:
                params.append(sub_module.lower_bound)
            if sub_module.upper_bound.requires_grad:
                params.append(sub_module.upper_bound)
    return params

def set_quant_requires_grad(module, enable=True):
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, FakeQuantizerBase):
            sub_module.lower_bound.requires_grad = enable
            sub_module.upper_bound.requires_grad = enable