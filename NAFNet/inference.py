import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.NAFNet_arch import NAFNet
from dataset import Mayo2dDataset
from torchvision.utils import save_image

def load_tiff(path):
    img = Image.open(path)
    img_array = np.array(img, dtype=np.float32)
    return img_array

def collect_test_data(input_dir):
    input_dir = Path(input_dir)
    files = sorted(list(input_dir.glob("*.tiff")) + list(input_dir.glob("*.tif")))
    if len(files) > 0:
         return [{'subfolder': '', 'files': files}]

    subfolders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    test_data = []
    
    for subfolder in subfolders:
        tiff_files = sorted(list(subfolder.glob("*.tiff")) + list(subfolder.glob("*.tif")))
        if len(tiff_files) >= 1:
            test_data.append({
                'subfolder': subfolder.name,
                'files': tiff_files
            })
    return test_data

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

    test_data = collect_test_data(args.test_lq)
    print(f"Found {len(test_data)} test sequences/folders")

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
    
    for seq in test_data:
        subfolder = seq['subfolder']
        files = seq['files']
        
        output_subfolder = os.path.join(args.output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        for i in tqdm(range(len(files)), desc=f"Processing {subfolder}"):
            # Boundaries
            if i == 0:
                idx_prev, idx_curr, idx_next = 0, 0, min(1, len(files)-1)
            elif i == len(files) - 1:
                idx_prev, idx_curr, idx_next = max(0, i-1), i, i
            else:
                idx_prev, idx_curr, idx_next = i-1, i, i+1
            
            img_prev = load_tiff(files[idx_prev])
            img_curr = load_tiff(files[idx_curr])
            img_next = load_tiff(files[idx_next])
            
            # Stack (1, 3, H, W)
            if img_curr.ndim == 2:
                img_stack = np.stack([img_prev, img_curr, img_next], axis=0)
            else:
                    img_stack = np.stack([img_prev, img_curr, img_next], axis=0)
                    if img_stack.ndim == 4 and img_stack.shape[3] == 1:
                        img_stack = img_stack.squeeze(3)

            img_tensor = torch.from_numpy(img_stack).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
            
            # Save
            out_img = output.squeeze().cpu().numpy()  
            save_path = os.path.join(output_subfolder, files[i].name)
            Image.fromarray(out_img).save(save_path)
            
    print("Inference Finished.")

if __name__ == '__main__':
    main()
