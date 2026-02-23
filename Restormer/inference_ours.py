import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.restormer_arch import Restormer
from dataset import Mayo2dDataset
from ours_quant import convert_to_quantized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gt', type=str, default='/data1/uni/data/mayo2d/test/full_1mm')
    parser.add_argument('--test_lq', type=str, default='/data1/uni/data/mayo2d/test/quarter_1mm')
    parser.add_argument('--model_path', type=str, required=True, help='Path to quantized model')
    parser.add_argument('--bit', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/data1/uni/data/mayo2d/test/out/Restormer_ours_output')
    parser.add_argument('--gpu', type=str, default='0')
    
    # Model args
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--enc_blk_nums', type=int, nargs='+', default=[1, 1, 1, 28])
    parser.add_argument('--middle_blk_num', type=int, default=1)
    parser.add_argument('--dec_blk_nums', type=int, nargs='+', default=[1, 1, 1, 1])

    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Dataset
    test_dataset = Mayo2dDataset(args.test_gt, args.test_lq, patch_size=0, is_train=False, n_patches=1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = Restormer(
        inp_channels=3,
        out_channels=1,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='BiasFree',
        dual_pixel_task=False
    )

    # Convert to quantized structure
    model = convert_to_quantized(model, args.bit)
    
    # -------------------------------------------------------------------
    # First and Last layers are always quantized to 8-bit
    # -------------------------------------------------------------------
    print(f"Applying Mixed Precision: Setting patch_embed.proj and output to 8-bit...")
    
    # 1. patch_embed.proj
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        if hasattr(model.patch_embed.proj, 'weight_quantizer'):
            model.patch_embed.proj.weight_quantizer.set_n_bit_manually(8)
            print("  > patch_embed.proj.weight_quantizer -> 8-bit")
        if hasattr(model.patch_embed.proj, 'act_quantizer'):
            model.patch_embed.proj.act_quantizer.set_n_bit_manually(8)
            print("  > patch_embed.proj.act_quantizer    -> 8-bit")
            
    # 2. output
    if hasattr(model, 'output'):
        if hasattr(model.output, 'weight_quantizer'):
            model.output.weight_quantizer.set_n_bit_manually(8)
            print("  > output.weight_quantizer           -> 8-bit")
        if hasattr(model.output, 'act_quantizer'):
            model.output.act_quantizer.set_n_bit_manually(8)
            print("  > output.act_quantizer            -> 8-bit")

    print(f"Loading quantized model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict) 
    
    model = model.to(device)

    for name, module in model.named_modules():
        if hasattr(module, 'calibrated'):
            module.calibrated = True
            
    model.eval()

    print(f"Start Ours Inference on {len(test_dataset)} frames with {args.bit}-bit model...")
    
    with torch.no_grad():
        for i, (lq, gt) in enumerate(tqdm(test_loader)):
            lq = lq.to(device)
            output = model(lq)
            
            out_img = output.squeeze().cpu().numpy()

            frame_info = test_dataset.frame_list[i]
            pair_idx = frame_info['pair_idx']
            frame_idx = frame_info['frame_idx']
            pair = test_dataset.data_pairs[pair_idx]
            original_file = pair['lq_files'][frame_idx]
            subfolder_name = pair['subfolder']
            filename = original_file.name

            save_subfolder = os.path.join(out_dir, subfolder_name)
            os.makedirs(save_subfolder, exist_ok=True)
            
            save_path = os.path.join(save_subfolder, filename)
            
            Image.fromarray(out_img).save(save_path)

    print(f"Inference done. Results saved to {out_dir}")

if __name__ == '__main__':
    main()
