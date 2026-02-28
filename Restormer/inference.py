import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import sys
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.restormer_arch import Restormer

# no paddings needed for MAyo, but needed for Moving700 due to Restormer's architecture.
# def pad_image(img_tensor, factor=8):
#     _, _, h, w = img_tensor.shape
    
#     pad_h = (factor - h % factor) % factor
#     pad_w = (factor - w % factor) % factor
    
#     if pad_h > 0 or pad_w > 0:
#         img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
#     return img_tensor, (h, w)

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
    parser.add_argument('--input_dir', default='<DATA_ROOT>/mayo2d/test/quarter_1mm', type=str)
    parser.add_argument('--result_dir', default='<DATA_ROOT>/mayo2d/test/out/Restormer_test', type=str)
    parser.add_argument('--weights', default='experiments/models/net_g_latest.pth', type=str)
    parser.add_argument('--gpu', default='2', type=str)
    args = parser.parse_args()
    
    # Restormer Args
    parser.add_argument('--inp_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--num_blocks', type=int, nargs='+', default=[4, 6, 6, 8])
    parser.add_argument('--num_refinement_blocks', type=int, default=4)
    parser.add_argument('--heads', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--ffn_expansion_factor', type=float, default=2.66)
    parser.add_argument('--bias', type=bool, default=False)

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.result_dir, exist_ok=True)

    model = Restormer(
        inp_channels=args.inp_channels,
        out_channels=args.out_channels,
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refinement_blocks,
        heads=args.heads,
        ffn_expansion_factor=args.ffn_expansion_factor,
        bias=args.bias,
        LayerNorm_type='BiasFree'
    ).to(device)
    
    
    checkpoint = torch.load(args.weights)
    if 'params_ema' in checkpoint:
        print("===> Loading EMA weights")
        model.load_state_dict(checkpoint['params_ema'])
    elif 'params' in checkpoint:
        print("===> Loading regular weights")
        model.load_state_dict(checkpoint['params'])
    else:
        print("===> Loading checkpoint directly")
        model.load_state_dict(checkpoint)

    model.eval()
    
    test_data = collect_test_data(args.input_dir)
    print(f"Found {len(test_data)} test sequences/folders")
    
    for seq in test_data:
        subfolder = seq['subfolder']
        files = seq['files']
        
        output_subfolder = os.path.join(args.result_dir, subfolder)
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

if __name__ == '__main__':
    main()
    