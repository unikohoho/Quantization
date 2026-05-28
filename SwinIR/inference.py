import os
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
try:
    from models.network_swinir import SwinIR
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from models.network_swinir import SwinIR

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
    parser.add_argument('--result_dir', default='<DATA_ROOT>/mayo2d/test/out/SwinIR', type=str)
    parser.add_argument('--weights', default='experiments/model_latest.pth', type=str)
    parser.add_argument('--gpu', default='2', type=str)
    args = parser.parse_args()
    
    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    
    print(f"Loading weights from {args.weights}")
    
    model = SwinIR(
        img_size=256,
        patch_size=1,
        in_chans=3,
        out_chans=1,
        embed_dim=60,
        depths=[6, 6, 6, 6],
        num_heads=[6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=torch.nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False, 
        upscale=1, 
        img_range=1., 
        upsampler='', 
        resi_connection='1conv'
    ).to(device)
    
    if os.path.exists(args.weights):
        try:
            model.load_state_dict(torch.load(args.weights, map_location=device))
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Weights loading warning: {e}")
            try:
                # Retry with strict=False
                model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
                print("Weights loaded with strict=False")
            except Exception as e2:
                 print(f"Weights loading failed: {e2}")
    else:
        print(f"Warning: Weight file {args.weights} not found. Running with random weights.")

    model.eval()
    
    test_data = collect_test_data(args.input_dir)
    print(f"Found {len(test_data)} test sequences/folders")
    
    for seq in test_data:
        subfolder = seq['subfolder']
        files = seq['files']
        
        output_subfolder = os.path.join(args.result_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)
        
        for i in tqdm(range(len(files)), desc=f"Processing {subfolder}"):
            # Boundary conditions (repeat edges)
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
                # SwinIR handles padding internally
                output = model(img_tensor)
            
            # Save
            out_img = output.squeeze().cpu().numpy()
                        
            save_path = os.path.join(output_subfolder, files[i].name)
             # Use generic TIFF saving if input was TIFF. PIL handles compression if needed.
            Image.fromarray(out_img).save(save_path)

if __name__ == '__main__':
    main()
