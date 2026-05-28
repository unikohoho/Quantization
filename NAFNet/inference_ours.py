import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys
from PIL import Image

# Ensure we can find basicsr
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.NAFNet_arch import NAFNet
from ours_quant import convert_to_quantized

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, lq_root):
        super(InferenceDataset, self).__init__()
        self.lq_root = Path(lq_root)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        subfolders = sorted([d for d in self.lq_root.iterdir() if d.is_dir()])
        
        for subfolder in subfolders:
            files = sorted(list(subfolder.glob("*.tiff")) + list(subfolder.glob("*.tif")))
            
            for i in range(1, len(files) - 1):
                triplet = (files[i-1], files[i], files[i+1])
                samples.append({
                    'paths': triplet,
                    'subfolder': subfolder.name,
                    'filename': files[i].name
                })
        return samples

    def _load_img(self, path):
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        return img

    def __getitem__(self, index):
        sample = self.samples[index]
        triplet = sample['paths']
        
        img_prev = self._load_img(triplet[0])
        img_curr = self._load_img(triplet[1])
        img_next = self._load_img(triplet[2])
        
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)
        lq_tensor = torch.from_numpy(np.ascontiguousarray(img_stack.transpose(2, 0, 1))).float()
        
        return lq_tensor, sample['subfolder'], sample['filename']

    def __len__(self):
        return len(self.samples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gt', type=str, default='')
    parser.add_argument('--test_lq', type=str, default='')
    parser.add_argument('--model_path', type=str, required=True, help='Path to quantized model')
    parser.add_argument('--bit', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    
    # Model args - must match training
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
    test_dataset = InferenceDataset(args.test_lq)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Model
    model = NAFNet(
        img_channel=3,
        out_channel=1,
        width=args.width,
        enc_blk_nums=args.enc_blk_nums,
        middle_blk_num=args.middle_blk_num,
        dec_blk_nums=args.dec_blk_nums
    )

    # Convert to quantized structure
    model = convert_to_quantized(model, args.bit)
    
    # -------------------------------------------------------------------
    # intro & ending layers should be 8-bit
    # -------------------------------------------------------------------
    print(f"Applying Mixed Precision: Setting intro and ending to 8-bit...")
    
    # 1. intro
    if hasattr(model, 'intro'):
        if hasattr(model.intro, 'weight_quantizer'):
            model.intro.weight_quantizer.set_n_bit_manually(8)
            print("  > intro.weight_quantizer -> 8-bit")
        if hasattr(model.intro, 'act_quantizer'):
            model.intro.act_quantizer.set_n_bit_manually(8)
            print("  > intro.act_quantizer    -> 8-bit")
            
    # 2. ending (Last Layer)
    if hasattr(model, 'ending'):
        if hasattr(model.ending, 'weight_quantizer'):
            model.ending.weight_quantizer.set_n_bit_manually(8)
            print("  > ending.weight_quantizer           -> 8-bit")
        if hasattr(model.ending, 'act_quantizer'):
            model.ending.act_quantizer.set_n_bit_manually(8)
            print("  > ending.act_quantizer            -> 8-bit")

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
        for i, (lq, subfolder, filename) in enumerate(tqdm(test_loader)):
            lq = lq.to(device)
            
            output = model(lq)            
            out_img = output.squeeze().cpu().numpy()
            
            subfolder_name = subfolder[0]
            fname = filename[0]

            save_subfolder = os.path.join(args.output_dir, subfolder_name)
            os.makedirs(save_subfolder, exist_ok=True)
            save_path = os.path.join(save_subfolder, fname)
            
            Image.fromarray(out_img).save(save_path)

    print(f"Inference done. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
