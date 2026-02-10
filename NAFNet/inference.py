import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

# Ensure we can find basicsr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from basicsr.models.archs.NAFNet_arch import NAFNet
from dataset import Mayo2dDataset
from torchvision.utils import save_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gt', type=str, default='/data1/uni/data/mayo2d/test/full_1mm')
    parser.add_argument('--test_lq', type=str, default='/data1/uni/data/mayo2d/test/quarter_1mm')
    parser.add_argument('--model_path', type=str, default='./experiments/models/model_latest.pth')
    parser.add_argument('--output_dir', type=str, default='/data1/uni/data/mayo2d/test/out/NAFNet')
    parser.add_argument('--gpu', type=str, default='2')
    
    # Model args - must match training
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--enc_blk_nums', type=int, nargs='+', default=[1, 1, 1, 28])
    parser.add_argument('--middle_blk_num', type=int, default=1)
    parser.add_argument('--dec_blk_nums', type=int, nargs='+', default=[1, 1, 1, 1])

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)

    test_dataset = Mayo2dDataset(args.test_gt, args.test_lq, patch_size=0, is_train=False, n_patches=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = NAFNet(
        img_channel=3,
        out_channel=1,
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums
    ).to(device)

    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Start Inference on {len(test_dataset)} frames...")
    
    with torch.no_grad():
        for i, (lq, gt) in enumerate(tqdm(test_loader)):
            lq = lq.to(device)
            
            # Forward
            output = model(lq)
            
            output = torch.clamp(output, 0, 1)
            
            out_img = output.squeeze().cpu().numpy() 
            
            current_file_idx = i  
            
            frame_info = test_dataset.frame_list[current_file_idx]
            pair_idx = frame_info['pair_idx']
            frame_idx = frame_info['frame_idx']
            
            pair = test_dataset.data_pairs[pair_idx]
            original_filename = pair['lq_files'][frame_idx].name
            subfolder_name = pair['subfolder']
            
            # Create subfolder in output_dir
            save_subdir = os.path.join(args.output_dir, subfolder_name)
            os.makedirs(save_subdir, exist_ok=True)
            
            save_path = os.path.join(save_subdir, original_filename)
            
            Image.fromarray(out_img).save(save_path)
            
    print("Inference Finished.")

if __name__ == '__main__':
    main()
