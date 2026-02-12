"""
Simple Percentile-based PTQ for Restormer
- Only percentile initialization (e.g., 0.999, 0.001)
- No calibration, no PAQ optimization
- Just clip outliers and apply uniform quantization
"""

import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.utils.options import parse
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.quant.convert2quant import convert_quantization


def percentile_ptq(opt_path):
    # Parse options
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False  # Single GPU for PTQ
    
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
    logger.info(f"Percentile-based PTQ for {opt['name']}")
    logger.info("=" * 60)
    
    # Create model
    model = create_model(opt)
    logger.info(f"Model created: {opt['model_type']}")
    logger.info(f"Quantization: {opt['q_config']['a_bit']}-bit activation, {opt['q_config']['w_bit']}-bit weight")
    logger.info(f"Percentile: {opt['q_config']['percentile']}")
    
    # Create calibration dataset (just for percentile initialization)
    logger.info("\n" + "=" * 60)
    logger.info("Stage: Percentile Initialization")
    logger.info("=" * 60)
    
    calib_set = create_dataset(opt['datasets']['stage1_dbdc'])
    
    # Add phase for dataloader
    calib_dataset_opt = opt['datasets']['stage1_dbdc'].copy()
    calib_dataset_opt['phase'] = 'train'  # Use train phase for dataloader
    
    calib_loader = create_dataloader(
        calib_set,
        calib_dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed']
    )
    
    calib_iter = opt['q_config'].get('calib_iter', 10)
    logger.info(f"Using {calib_iter} samples for percentile initialization")
    
    # Run forward passes to initialize percentile-based clip_val
    model.net_g.eval()
    logger.info("Initializing quantization ranges with percentile...")
    
    with torch.no_grad():
        for idx, calib_data in enumerate(calib_loader):
            if idx >= calib_iter:
                break
            
            # Forward pass - this will trigger percentile initialization in quantizers
            _ = model.net_g(calib_data['lq'].cuda())
            
            if (idx + 1) % 5 == 0:
                logger.info(f"  Processed {idx + 1}/{calib_iter} samples")
    
    logger.info("Percentile initialization complete!")
    
    # Disable calibration mode (freeze clip_val)
    from basicsr.quant.convert2quant import disable_calibration
    disable_calibration(model.net_g)
    logger.info("Calibration disabled - quantization ranges are now fixed")
    
    # Print some quantization statistics
    logger.info("\n" + "-" * 60)
    logger.info("Quantization Statistics")
    logger.info("-" * 60)
    count = 0
    for name, module in model.net_g.named_modules():
        if hasattr(module, 'wgt_quantizer'):
            wgt_q = module.wgt_quantizer
            act_q = module.act_quantizer
            logger.info(f"\n{name}:")
            
            # Handle both scalar and per-channel clip_val
            if wgt_q.lower_clip_val.numel() == 1:
                logger.info(f"  Weight: [{wgt_q.lower_clip_val.item():.4f}, {wgt_q.upper_clip_val.item():.4f}]")
            else:
                logger.info(f"  Weight (per-channel): min=[{wgt_q.lower_clip_val.min().item():.4f}], max=[{wgt_q.upper_clip_val.max().item():.4f}]")
            
            if act_q.lower_clip_val.numel() == 1:
                logger.info(f"  Activation: [{act_q.lower_clip_val.item():.4f}, {act_q.upper_clip_val.item():.4f}]")
            else:
                logger.info(f"  Activation (per-channel): min=[{act_q.lower_clip_val.min().item():.4f}], max=[{act_q.upper_clip_val.max().item():.4f}]")
            
            count += 1
            # if count >= 3:  # Show first 3 layers only
            #     logger.info("  ...")
            #     break
    print("Number of quantized layers: ", count)
    
    # Save quantized model
    save_dir = '../experiments/quant'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{opt['name']}_percentile.pth")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Saving quantized model to: {save_path}")
    torch.save(model.net_g.state_dict(), save_path)
    logger.info("Model saved successfully!")
    
    logger.info("\n" + "=" * 60)
    logger.info("Percentile-based PTQ completed!")
    logger.info("=" * 60)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    
    percentile_ptq(args.opt)
