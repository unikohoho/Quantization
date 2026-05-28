import torch.nn as nn
from .quant_arch import QuantConv2d, QuantLinear, FakeQuantizerBase, QuantLinearQKV, QuantConv2dQKV

def convert_to_quantized(module, bit, layer_name=""):
    # ---------------------------------------------------------
    # NAFNet 구조에 맞춘 Selective Channel-wise 로직
    # ---------------------------------------------------------
    is_cw_act = False 
    
    # 1. intro layer
    if layer_name == 'intro' or layer_name.startswith('intro.'):
        is_cw_act = True
        print(f"Applying Channel-wise Quantization to {layer_name}")
    elif '.sca.1' in layer_name:
        is_cw_act = True
        print(f"Applying Channel-wise Quantization to {layer_name}")
    elif 'decoders.2' in layer_name or 'decoders.3' in layer_name:
        is_cw_act = True
        print(f"Applying Channel-wise Quantization to {layer_name}")
    elif 'encoders.0' in layer_name or 'encoders.1' in layer_name:
        is_cw_act = True
        print(f"Applying Channel-wise Quantization to {layer_name}")

    config = {
        'bit': bit, 
        'channel_wise': is_cw_act,    
        'channel_wise_weight': False,  
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
        
        if name == 'qkv':
            if isinstance(submodule, nn.Linear):
                q_layer = QuantLinearQKV(config)
                q_layer.set_param(submodule)
                setattr(module, name, q_layer)
                continue
            elif isinstance(submodule, nn.Conv2d):
                q_layer = QuantConv2dQKV(config)
                q_layer.set_param(submodule)
                setattr(module, name, q_layer)
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