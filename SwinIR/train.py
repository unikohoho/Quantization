import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from models.network_swinir import SwinIR
except ImportError:
    # If running from inside SwinIR folder
    import sys
    sys.path.append(os.getcwd())
    from models.network_swinir import SwinIR

from dataset import Mayo2dDataset

def main():
    parser = argparse.ArgumentParser()
    # Path args
    parser.add_argument('--train_gt', type=str, default='/data1/uni/data/mayo2d/train/full_1mm')
    parser.add_argument('--train_lq', type=str, default='/data1/uni/data/mayo2d/train/quarter_1mm')
    # Use different paths for validation if needed, or same structure
    parser.add_argument('--val_gt', type=str, default='/data1/uni/data/mayo2d/test/full_1mm')
    parser.add_argument('--val_lq', type=str, default='/data1/uni/data/mayo2d/test/quarter_1mm')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=4) # Increased batch size for lightweight model
    parser.add_argument('--patch_size', type=int, default=256) # Matched gray_dn training patch size
    parser.add_argument('--n_patches', type=int, default=4, help='Number of patches to extract from one image')
    parser.add_argument('--epochs', type=int, default=30) # Increased epochs
    parser.add_argument('--lr', type=float, default=1e-3) # Standard lr for SwinIR
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='3')
    
    # Model args
    parser.add_argument('--embed_dim', type=int, default=60)
    parser.add_argument('--window_size', type=int, default=8)
    
    args = parser.parse_args()
    '''
    Epoch 1 Average Loss: 0.040128
    Epoch 2 Average Loss: 0.011618
    Epoch 3 Average Loss: 0.011090
    Epoch 4 Average Loss: 0.011094
    Epoch 20 Average Loss: 0.008860
    '''
    '''
    Normalization 부분 수정 후 재학습
    Epoch 1 Average Loss: 0.055570
    Epoch 2 Average Loss: 0.012431
    Epoch 3 Average Loss: 0.017457
    Epoch 4 Average Loss: 0.011395
    Epoch 5 Average Loss: 0.011183
    응안돼.. 
    '''
    '''
    모델 크기 줄여서 재학습 (lightweight) 
    Epoch 1 Average Loss: 0.034851
    Epoch 2 Average Loss: 0.014265 / -0.020586
    Epoch 3 Average Loss: 0.012038 / -0.002227
    Epoch 4 Average Loss: 0.009133 / -0.002905
    Epoch 5 Average Loss: 0.007900 / -0.001233
    Epoch 6 Average Loss: 0.007529 / -0.000371
    Epoch 7 Average Loss: 0.007245 / -0.000284
    Epoch 8 Average Loss: 0.007009 / -0.000236
    Epoch 9 Average Loss: 0.006888 / -0.000121
    Epoch 10 Average Loss: 0.006618 / -0.000270
    Epoch 11 Average Loss: 0.006401 / -0.000217
    Epoch 12 Average Loss: 0.005572 / -0.000829
    Epoch 13 Average Loss: 0.005007 / -0.000565
    Epoch 14 Average Loss: 0.004767 / -0.000240
    Epoch 15 Average Loss: 0.004723 / -0.000044
    Epoch 16 Average Loss: 0.004640 / -0.000083
    Epoch 17 Average Loss: 0.004599 / -0.000041
    Epoch 18 Average Loss: 0.004604 / +0.000005
    Epoch 19 Average Loss: 0.004595 / -0.000009
    Epoch 20 Average Loss: 0.004563 / -0.000032
    '''

    '''
    Mayo2d Dataset
    Epoch 1 Average Loss: 0.020735
    Epoch 2 Average Loss: 0.015309
    Epoch 3 Average Loss: 0.015278
    Epoch 4 Average Loss: 0.015215
    Epoch 5 Average Loss: 0.013338
    Epoch 6 Average Loss: 0.012648
    Epoch 7 Average Loss: 0.012586
    Epoch 8 Average Loss: 0.012586
    Epoch 9 Average Loss: 0.012862
    Epoch 10 Average Loss: 0.012493
    Epoch 11 Average Loss: 0.012449
    Epoch 12 Average Loss: 0.012438
    Epoch 13 Average Loss: 0.012391
    Epoch 14 Average Loss: 0.012363
    Epoch 15 Average Loss: 0.012347
    Epoch 16 Average Loss: 0.012352
    Epoch 17 Average Loss: 0.012339
    Epoch 18 Average Loss: 0.012316
    Epoch 19 Average Loss: 0.012313
    Epoch 20 Average Loss: 0.012289
    Epoch 21 Average Loss: 0.012275
    Epoch 22 Average Loss: 0.012271
    Epoch 23 Average Loss: 0.012277
    Epoch 24 Average Loss: 0.012248
    Epoch 25 Average Loss: 0.012239
    Epoch 26 Average Loss: 0.012257
    Epoch 27 Average Loss: 0.012259
    Epoch 28 Average Loss: 0.012229
    Epoch 29 Average Loss: 0.012234
    Epoch 30 Average Loss: 0.012235
    '''
        

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    print("Initializing Datasets...")
    train_dataset = Mayo2dDataset(args.train_gt, args.train_lq, patch_size=args.patch_size, is_train=True, n_patches=args.n_patches)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Model configuration for Denoising (upscale=1, upsampler='')
    print("Initializing SwinIR...")
    model = SwinIR(
        img_size=args.patch_size,
        patch_size=1,
        in_chans=3,
        out_chans=1,
        embed_dim=args.embed_dim,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=args.window_size,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False, 
        upscale=1, 
        img_range=1., 
        upsampler='',  # No upsampler for denoising
        resi_connection='1conv'
    ).to(device)
    
    # Params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # SwinIR paper uses L1 Loss/Charbonnier Loss
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
