import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Add current dir to path to find basicsr
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basicsr.models.archs.NAFNet_arch import NAFNet
from dataset import Mayo2dDataset

def main():
    parser = argparse.ArgumentParser()
    # Path args
    parser.add_argument('--train_gt', type=str, default='/data1/uni/data/mayo2d/train/full_1mm')
    parser.add_argument('--train_lq', type=str, default='/data1/uni/data/mayo2d/train/quarter_1mm')
    parser.add_argument('--val_gt', type=str, default='/data1/uni/data/mayo2d/test/full_1mm')
    parser.add_argument('--val_lq', type=str, default='/data1/uni/data/mayo2d/test/quarter_1mm')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--patch_size', type=int, default=256) 
    parser.add_argument('--n_patches', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./experiments/models')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='2')
    
    # Model args
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--enc_blk_nums', type=int, nargs='+', default=[1, 1, 1, 28])
    parser.add_argument('--middle_blk_num', type=int, default=1)
    parser.add_argument('--dec_blk_nums', type=int, nargs='+', default=[1, 1, 1, 1])
    
    args = parser.parse_args()
    '''
    Epoch 1 Average Loss: 0.015064
    Epoch 2 Average Loss: 0.012887, -0.002177
    Epoch 3 Average Loss: 0.012616, -0.000271
    Epoch 4 Average Loss: 0.012501, -0.000115
    Epoch 5 Average Loss: 0.012421, -0.000080
    Epoch 6 Average Loss: 0.012388, -0.000033
    Epoch 7 Average Loss: 0.012336, -0.000052
    Epoch 8 Average Loss: 0.012281, -0.000055
    Epoch 9 Average Loss: 0.012254, -0.000027
    Epoch 10 Average Loss: 0.012192, -0.000062
    Epoch 11 Average Loss: 0.012208, +0.000016
    ...
    Epoch 30 Average Loss: 0.012003
    '''

    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    print("Initializing Datasets...")
    train_dataset = Mayo2dDataset(args.train_gt, args.train_lq, patch_size=args.patch_size, is_train=True, n_patches=args.n_patches)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Model configuration
    print("Initializing NAFNet...")
    # 3 frames input, 1 frame output
    model = NAFNet(
        img_channel=3,
        out_channel=1,
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums
    ).to(device)
    
    # Params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Loss: PSNR-oriented -> L1 or Charbonnier
    criterion = nn.L1Loss()
    
    # Training Loop
    print("Start Training...")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        epoch_loss = 0.0
        
        for lq, gt in pbar:
            lq = lq.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            output = model(lq)
            
            loss = criterion(output, gt)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {step}. Skipping batch.")
                optimizer.zero_grad()
                continue
                
            loss.backward()
            
            # Clip grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01) # Check if needed for NAFNet
            
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            step += 1
            
        print(f"Epoch {epoch} Average Loss: {epoch_loss / len(train_loader):.6f}")
        
        # Save
        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")
        
        scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

if __name__ == '__main__':
    main()
