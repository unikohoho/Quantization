"""
PTQ4SR adapted for Restormer Multi-frame CT Denoising
"""

import logging
import torch
import torch.nn as nn
from os import path as osp
from functools import partial
import numpy as np
import torch.nn.functional as F

from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse


def create_quant_dataloader(opt):
    """Create calibration and PAQ dataloaders"""
    calib_train_loader, paq_train_loader = None, None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'stage1_dbdc':
            dataset_opt['phase'] = 'train'
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            calib_train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
        elif phase == 'stage2_paq':
            dataset_opt['phase'] = 'train'
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            paq_train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])

    return calib_train_loader, paq_train_loader


def collect_quantization_stats(model, logger):
    """
    Collect and log detailed quantization statistics for each layer
    """
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED QUANTIZATION STATISTICS")
    logger.info("=" * 80)
    
    stats = {
        'total_modules': 0,  # All modules
        'conv_linear_layers': 0,  # Conv2d + Linear only
        'quantized_layers': 0,
        'fp32_layers': 0,
        'weight_quant': [],
        'activation_quant': []
    }
    
    for name, module in model.named_modules():
        # Count all modules
        if len(list(module.children())) == 0:  # Only leaf modules
            stats['total_modules'] += 1
        
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            stats['conv_linear_layers'] += 1
            
            # Check if layer is quantized
            if hasattr(module, 'w_bit') and hasattr(module, 'a_bit'):
                if module.w_bit != 32 or module.a_bit != 32:
                    stats['quantized_layers'] += 1
                    
                    # Collect weight quantization info
                    if module.w_bit != 32 and hasattr(module, 'wgt_quantizer'):
                        wgt_q = module.wgt_quantizer
                        if hasattr(wgt_q, 'lower_clip_val') and hasattr(wgt_q, 'upper_clip_val'):
                            lower = wgt_q.lower_clip_val.cpu().detach()
                            upper = wgt_q.upper_clip_val.cpu().detach()
                            
                            # Calculate actual weight range
                            w_min = module.weight.min().item()
                            w_max = module.weight.max().item()
                            w_mean = module.weight.mean().item()
                            w_std = module.weight.std().item()
                            
                            # Calculate clipping ratio
                            if wgt_q.per_channel:
                                # For per-channel, reshape lower/upper to match weight dimensions
                                total_elements = module.weight.numel()
                                weight_cpu = module.weight.cpu().detach()
                                # Reshape to [out_channels, 1, 1, ...] for broadcasting
                                lower_reshaped = lower.view(-1, *([1] * (weight_cpu.dim() - 1)))
                                upper_reshaped = upper.view(-1, *([1] * (weight_cpu.dim() - 1)))
                                clipped_elements = ((weight_cpu < lower_reshaped) | (weight_cpu > upper_reshaped)).sum().item()
                                clip_ratio = clipped_elements / total_elements * 100
                            else:
                                # For per-tensor, direct comparison
                                total_elements = module.weight.numel()
                                weight_cpu = module.weight.cpu().detach()
                                clipped_elements = ((weight_cpu < lower) | (weight_cpu > upper)).sum().item()
                                clip_ratio = clipped_elements / total_elements * 100
                            
                            if wgt_q.per_channel:
                                logger.info(f"\nLayer: {name}")
                                logger.info(f"  Type: {type(module).__name__}")
                                logger.info(f"  Weight: {module.w_bit}-bit (per-channel)")
                                logger.info(f"  Weight range: [{w_min:.6f}, {w_max:.6f}], mean: {w_mean:.6f}, std: {w_std:.6f}")
                                logger.info(f"  Clipping range: per-channel ({lower.numel()} channels)")
                                logger.info(f"  Clipped values: {clipped_elements}/{total_elements} ({clip_ratio:.4f}%)")
                            else:
                                lower_val = lower.item() if lower.numel() == 1 else lower.mean().item()
                                upper_val = upper.item() if upper.numel() == 1 else upper.mean().item()
                                logger.info(f"\nLayer: {name}")
                                logger.info(f"  Type: {type(module).__name__}")
                                logger.info(f"  Weight: {module.w_bit}-bit (per-tensor)")
                                logger.info(f"  Weight range: [{w_min:.6f}, {w_max:.6f}], mean: {w_mean:.6f}, std: {w_std:.6f}")
                                logger.info(f"  Quantization range: [{lower_val:.6f}, {upper_val:.6f}]")
                                logger.info(f"  Clipped values: {clipped_elements}/{total_elements} ({clip_ratio:.4f}%)")
                                logger.info(f"  Quantization levels: {2**module.w_bit}")
                                logger.info(f"  Step size: {(upper_val - lower_val) / (2**module.w_bit - 1):.8f}")
                            
                            lower_val = lower.item() if lower.numel() == 1 else 'per_channel'
                            upper_val = upper.item() if upper.numel() == 1 else 'per_channel'
                            stats['weight_quant'].append({
                                'layer': name,
                                'bits': module.w_bit,
                                'lower': lower_val,
                                'upper': upper_val,
                                'clip_ratio': clip_ratio
                            })
                    
                    # Collect activation quantization info
                    if module.a_bit != 32 and hasattr(module, 'act_quantizer'):
                        act_q = module.act_quantizer
                        if hasattr(act_q, 'lower_clip_val') and hasattr(act_q, 'upper_clip_val'):
                            lower = act_q.lower_clip_val.cpu().detach()
                            upper = act_q.upper_clip_val.cpu().detach()
                            lower_val = lower.item() if lower.numel() == 1 else lower.mean().item()
                            upper_val = upper.item() if upper.numel() == 1 else upper.mean().item()
                            
                            logger.info(f"  Activation: {module.a_bit}-bit")
                            logger.info(f"  Activation quantization range: [{lower_val:.6f}, {upper_val:.6f}]")
                            logger.info(f"  Activation quantization levels: {2**module.a_bit}")
                            logger.info(f"  Activation step size: {(upper_val - lower_val) / (2**module.a_bit - 1):.8f}")
                            
                            stats['activation_quant'].append({
                                'layer': name,
                                'bits': module.a_bit,
                                'lower': lower_val,
                                'upper': upper_val
                            })
                else:
                    stats['fp32_layers'] += 1
                    # Log important FP32 layers (first and last layers)
                    if 'patch_embed.proj' in name or 'output' in name:
                        logger.info(f"\nLayer: {name}")
                        logger.info(f"  Type: {type(module).__name__}")
                        logger.info(f"  Precision: FP32 (not quantized) ⚠️ IMPORTANT: {'First layer' if 'patch_embed.proj' in name else 'Last layer'}")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total modules (leaf nodes): {stats['total_modules']}")
    logger.info(f"Conv2d/Linear layers: {stats['conv_linear_layers']}")
    logger.info(f"Quantized layers: {stats['quantized_layers']}")
    logger.info(f"FP32 layers: {stats['fp32_layers']}")
    if stats['conv_linear_layers'] > 0:
        logger.info(f"Quantization ratio (Conv2d/Linear): {stats['quantized_layers']/stats['conv_linear_layers']*100:.2f}%")
    
    if stats['weight_quant']:
        logger.info(f"\nWeight quantization:")
        logger.info(f"  Average bits: {np.mean([s['bits'] for s in stats['weight_quant']]):.2f}")
        valid_clip_ratios = [s['clip_ratio'] for s in stats['weight_quant'] if isinstance(s['clip_ratio'], (int, float))]
        if valid_clip_ratios:
            logger.info(f"  Average clip ratio: {np.mean(valid_clip_ratios):.4f}%")
    
    if stats['activation_quant']:
        logger.info(f"\nActivation quantization:")
        logger.info(f"  Average bits: {np.mean([s['bits'] for s in stats['activation_quant']]):.2f}")
        logger.info(f"  Average range: [{np.mean([s['lower'] for s in stats['activation_quant']]):.4f}, "
                   f"{np.mean([s['upper'] for s in stats['activation_quant']]):.4f}]")
    
    logger.info("=" * 80)
    
    return stats



def test_pipeline(opt_path):
    """Main PTQ pipeline for Restormer"""
    # Parse options
    opt = parse(opt_path, is_train=False)
    
    # Set distributed training to False for PTQ
    opt['dist'] = False

    torch.backends.cudnn.benchmark = True

    # Initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    
    # Add console handler to ensure logs are printed to stdout as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file: {log_file}")
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if "test" in phase:
            test_set = create_dataset(dataset_opt)
            test_loader = create_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)

    # Create model
    model = create_model(opt)

    # Create calibration and PAQ dataloaders
    logger.info("Calibration and finetune dataset preparing...")
    calib_train_loader, paq_train_loader = create_quant_dataloader(opt)
    
    from basicsr.quant.convert2quant import disable_calibration, print_range

    # Stage 1: Calibration
    logger.info("=" * 60)
    logger.info("Stage 1: Calibration")
    logger.info("=" * 60)
    logger.info(f"Calibration iterations: {model.q_config['calib_iter']}")
    logger.info(f"Weight bit-width: {model.q_config['w_bit']}")
    logger.info(f"Activation bit-width: {model.q_config['a_bit']}")
    logger.info(f"Weight initialization: {model.q_config['w_init']}")
    logger.info(f"Activation initialization: {model.q_config['a_init']}")
    logger.info("-" * 60)
    
    for iter, train_data in enumerate(calib_train_loader):
        logger.info(f"Calibrate ({iter+1}/{model.q_config['calib_iter']})")
        model.calibration(train_data)
        if iter+1 == model.q_config['calib_iter']:
            break

    model.calibrate = False
    model.model_quant = True

    if model.quant:
        disable_calibration(model.net_g)
        logger.info("\n" + "=" * 80)
        logger.info("QUANTIZATION RANGE AFTER CALIBRATION")
        logger.info("=" * 80)
        print_range(model.net_g, logger)
        logger.info("=" * 80)
        
        # Collect detailed statistics
        calib_stats = collect_quantization_stats(model.net_g, logger)

    # Skip validation after calibration to save time
    # logger.info("\n" + "=" * 60)
    # logger.info("Validation after calibration")
    # logger.info("=" * 60)
    # for test_loader in test_loaders[:1]:  # Test on first dataset only for speed
    #     test_set_name = test_loader.dataset.opt['name']
    #     logger.info(f'Testing {test_set_name}...')
    #     model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    # Stage 2: PAQ (Post-training Quantization-aware) optimization
    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: PAQ Optimization")
    logger.info("=" * 60)
    
    qparams_w, qparams_x = [], []
    for name, param in model.net_g.named_parameters():
        if "wgt_quantizer" in name:
            qparams_w.append(param)
        elif "act_quantizer" in name:
            qparams_x.append(param)

    model.net_g.train()

    logger.info(f"Quantization parameters: qparams_w={len(qparams_w)}, qparams_x={len(qparams_x)}")

    paq_w_opt_iter = model.q_config['paq_w_opt_iter']
    paq_a_opt_iter = model.q_config['paq_a_opt_iter']

    opt_qparams_w = torch.optim.Adam(qparams_w, lr=model.q_config['opt_w_lr'][0])
    opt_qparams_x = torch.optim.Adam(qparams_x, lr=model.q_config['opt_a_lr'][0])
    scheduler_qparams_w = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_qparams_w, T_max=paq_w_opt_iter, eta_min=model.q_config['opt_w_lr'][1])
    scheduler_qparams_x = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_qparams_x, T_max=paq_a_opt_iter, eta_min=model.q_config['opt_a_lr'][1])

    # For Restormer, use TransformerBlock instead of ResidualBlockNoBN
    from basicsr.models.archs.restormer_arch import TransformerBlock
    from basicsr.quant.layers import feature_loss
    
    module_names, fp_modules, quant_modules = [], [], []
    for name, module in model.net_g.named_modules():
        if isinstance(module, TransformerBlock):
            module_names.append(name)
            quant_modules.append(module)
    for name, module in model.net_fp.named_modules():
        if isinstance(module, TransformerBlock):
            fp_modules.append(module)
    
    logger.info(f"Feature matching modules: names={len(module_names)}, fp_modules={len(fp_modules)}, quant_modules={len(quant_modules)}")

    losses_w, losses_w_out, losses_w_fea = [], [], []
    losses_x, losses_x_out, losses_x_fea = [], [], []
    factor = 5  # Feature loss weight

    # Insert hooks for feature matching
    fp32_output = []
    def hook(name, module, input, output):
        fp32_output.append(output.detach())  # Keep on GPU, just detach

    quant_output = []
    def Qhook(name, module, input, output):
        quant_output.append(output)  # Keep gradient, keep on GPU

    fp_handle_list, quant_handle_list = [], []
    for idx, module_name in enumerate(module_names):
        fp_handle_list.append(fp_modules[idx].register_forward_hook(partial(hook, module_name)))
        quant_handle_list.append(quant_modules[idx].register_forward_hook(partial(Qhook, module_name)))

    # Optimize activation quantization parameters
    logger.info("\n" + "-" * 60)
    logger.info("Optimizing Activation Quantization Parameters")
    logger.info("-" * 60)
    logger.info(f"Total iterations: {paq_a_opt_iter}")
    logger.info(f"Learning rate: {model.q_config['opt_a_lr'][0]} → {model.q_config['opt_a_lr'][1]}")
    logger.info(f"Feature matching factor: {factor}")
    logger.info("-" * 60)
    
    for iter, train_data in enumerate(paq_train_loader):
        # Forward passes
        out_fp = model.fp_inference(train_data)
        out_quant = model.quant_inference(train_data)

        loss_out = F.l1_loss(out_fp, out_quant)

        fea_loss = 0
        for layer_idx in range(len(fp32_output)):
            fea_loss += feature_loss(fp32_output[layer_idx], quant_output[layer_idx])
        fea_loss = fea_loss / len(fp32_output)

        loss = loss_out + factor * fea_loss
        
        # Clear feature lists before backward to save memory
        fp32_output.clear()
        quant_output.clear()

        losses_x.append(loss.item())
        losses_x_out.append(loss_out.item())
        losses_x_fea.append(factor * fea_loss.item())
        
        opt_qparams_x.zero_grad()
        loss.backward()
        opt_qparams_x.step()

        if (iter + 1) % 10 == 0 or iter == 0:
            logger.info(f"Optimize activation, iter: {iter+1:4d}/{paq_a_opt_iter}, "
                       f"lr_x: {opt_qparams_x.param_groups[0]['lr']:.7f}, "
                       f"cur loss: {loss:.6f}, avg loss: {np.mean(losses_x):.6f}, "
                       f"avg out loss: {np.mean(losses_x_out):.6f}, "
                       f"avg fea loss: {np.mean(losses_x_fea):.6f}")
        
        if iter == paq_a_opt_iter - 1:
            break
        
        scheduler_qparams_x.step()
        
        # Clear CUDA cache periodically
        if (iter + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    logger.info("-" * 60)
    logger.info(f"Activation optimization completed")
    logger.info(f"Final average loss: {np.mean(losses_x):.6f}")
    logger.info("-" * 60)

    # Optimize weight quantization parameters
    logger.info("\n" + "-" * 60)
    logger.info("Optimizing Weight Quantization Parameters")
    logger.info("-" * 60)
    logger.info(f"Total iterations: {paq_w_opt_iter}")
    logger.info(f"Learning rate: {model.q_config['opt_w_lr'][0]} → {model.q_config['opt_w_lr'][1]}")
    logger.info(f"Feature matching factor: {factor}")
    logger.info("-" * 60)
    
    for iter, train_data in enumerate(paq_train_loader):
        out_fp = model.fp_inference(train_data)
        out_quant = model.quant_inference(train_data)

        loss_out = F.l1_loss(out_fp, out_quant)

        fea_loss = 0
        for layer_idx in range(len(fp32_output)):
            fea_loss += feature_loss(fp32_output[layer_idx], quant_output[layer_idx])
        fea_loss = fea_loss / len(fp32_output)

        loss = loss_out + factor * fea_loss
        
        # Clear feature lists before backward to save memory
        fp32_output.clear()
        quant_output.clear()

        losses_w.append(loss.item())
        losses_w_out.append(loss_out.item())
        losses_w_fea.append(factor * fea_loss.item())
        
        opt_qparams_w.zero_grad()
        loss.backward()
        opt_qparams_w.step()

        if (iter + 1) % 10 == 0 or iter == 0:
            logger.info(f"Optimize weight, iter: {iter+1:4d}/{paq_w_opt_iter}, "
                       f"lr_w: {opt_qparams_w.param_groups[0]['lr']:.7f}, "
                       f"cur loss: {loss:.6f}, avg loss: {np.mean(losses_w):.6f}, "
                       f"avg out loss: {np.mean(losses_w_out):.6f}, "
                       f"avg fea loss: {np.mean(losses_w_fea):.6f}")
        
        if iter == paq_w_opt_iter - 1:
            break

        scheduler_qparams_w.step()
        
        # Clear CUDA cache periodically
        if (iter + 1) % 50 == 0:
            torch.cuda.empty_cache()
    
    logger.info("-" * 60)
    logger.info(f"Weight optimization completed")
    logger.info(f"Final average loss: {np.mean(losses_w):.6f}")
    logger.info("-" * 60)

    # Remove hooks
    for handle in fp_handle_list:
        handle.remove()
    for handle in quant_handle_list:
        handle.remove()

    # Collect statistics after PAQ optimization
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZATION STATISTICS AFTER PAQ OPTIMIZATION")
    logger.info("=" * 80)
    paq_stats = collect_quantization_stats(model.net_g, logger)

    # Final validation
    logger.info("\n" + "=" * 60)
    logger.info("Final Validation Skipped")
    logger.info("=" * 60)

    # logger.info("Final Validation after PAQ")
    # logger.info("=" * 60)
    # for test_loader in test_loaders:
    #     test_set_name = test_loader.dataset.opt['name']
    #     logger.info(f'Testing {test_set_name}...')
    #     model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    # Save quantized model
    save_path = f"./experiments/quant/{opt['name']}.pth"
    
    # Create directory if not exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(model.net_g.state_dict(), save_path)
    logger.info(f"\nQuantized model saved to: {save_path}")


if __name__ == '__main__':
    import sys
    # Get the config file from command line arguments
    if len(sys.argv) > 2 and sys.argv[1] == '-opt':
        opt_path = sys.argv[2]
    else:
        raise ValueError("Usage: python ptq4sr_restormer.py -opt <config_file>")
    
    test_pipeline(opt_path)
