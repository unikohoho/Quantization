import os
import argparse
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.network_swinir import SwinIR
from ours_quant import convert_to_quantized
from ours_quant.convs.wavelet import SWTForward, serialize_swt
from ours_quant.convs.norm import Norm2d_

class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, lq_root):
        super(CalibrationDataset, self).__init__()
        self.lq_root = Path(lq_root)
        
        self.fixed_seed = random.Random(21)
        
        self.file_list = self._collect_one_sample_per_folder()
        
    def _collect_one_sample_per_folder(self):
        sample_list = []
        subfolders = sorted([d for d in self.lq_root.iterdir() if d.is_dir()])
        samples_per_folder = 10
        
        for subfolder in subfolders:
            files = sorted(list(subfolder.glob("*.tiff")) + list(subfolder.glob("*.tif")))
            
            if len(files) < 3:
                continue
                
            valid_indices = list(range(1, len(files) - 1))
            num_samples = min(len(valid_indices), samples_per_folder)
            
            selected_indices = self.fixed_seed.sample(valid_indices, num_samples)
            
            for idx in selected_indices:
                triplet = (files[idx-1], files[idx], files[idx+1])
                sample_list.append(triplet)
        
        print(f"Calibration Dataset: Selected {len(sample_list)} samples from {len(subfolders)} subfolders.")
        return sample_list

    def _load_img(self, path):
        img = Image.open(path)
        img = np.array(img, dtype=np.float32) 
        return img

    def __getitem__(self, index):
        path_prev, path_curr, path_next = self.file_list[index]
        
        img_prev = self._load_img(path_prev)
        img_curr = self._load_img(path_curr)
        img_next = self._load_img(path_next)
        
        img_stack = np.stack([img_prev, img_curr, img_next], axis=-1)
        lq_tensor = torch.from_numpy(np.ascontiguousarray(img_stack.transpose(2, 0, 1))).float()
        
        # GT는 dummy
        gt_tensor = torch.zeros((1, lq_tensor.shape[1], lq_tensor.shape[2]))
        
        return lq_tensor, gt_tensor

    def __len__(self):
        return len(self.file_list)

# -------------------------------------------------------------
# [Data Processing 1] Center Crop 
# -------------------------------------------------------------
def get_center_crop(img_tensor, patch_size=128):
    b, c, h, w = img_tensor.shape
    start_h = max(0, (h - patch_size) // 2)
    start_w = max(0, (w - patch_size) // 2)
    return img_tensor[:, :, start_h:start_h+patch_size, start_w:start_w+patch_size]

# -------------------------------------------------------------
# [Data Processing 2] Random Crop & Augmentation 
# -------------------------------------------------------------
def get_random_patches_with_aug(img_tensor, patch_size=128):
    b, c, h, w = img_tensor.shape
    patches = []
    
    for i in range(b):
        img = img_tensor[i] 
        
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)

        patch = img[:, top:top+patch_size, left:left+patch_size]
        
        if random.random() < 0.5:
            patch = torch.flip(patch, [2]) 
            
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            patch = torch.rot90(patch, k=rot_k, dims=[1, 2])
            
        patches.append(patch)
        
    return torch.stack(patches, dim=0) 

class OptimizationWrapper:
    def __init__(self, quant_model, fp_model, dataloader_small, dataloader_large, device):
        self.net_Q = quant_model
        self.net_F = fp_model
        self.dataloader_small = dataloader_small
        self.dataloader_large = dataloader_large
        self.device = device
        
        self.swt = SWTForward(J=2, wave='db1', mode='periodic').to(device) 
        self.swt_norm = Norm2d_(7).to(device) 

        self.schedule = []
        for _ in range(2):
            self.schedule.append(('both', 1000))
            self.schedule.append(('act', 1000))
        self.schedule.append(('both', 1000))


        self.total_act_steps = sum(steps for p, steps in self.schedule if p in ['act', 'both'])
        self.total_weight_steps = sum(steps for p, steps in self.schedule if p in ['weight', 'both'])
        
    def optimize(self):
        print("Starting Optimization Process...")
        self.net_Q.eval()
        self.net_F.eval()

        # -------------------------------------------------------
        # [Step 1] Initialization with Center Crop 
        # -------------------------------------------------------
        print("\n[Step 1] Initializing Bounds with MSE (Center Crop - Large Batch)...")
        for name, module in self.net_Q.named_modules():
             if hasattr(module, 'calibrated'):
                 module.calibrated = False 
                 if hasattr(module, 'init_done'): module.init_done = False

        with torch.no_grad():
            for i, (lq, _) in enumerate(self.dataloader_large):
                print(f"  > Processing Batch {i+1}/{len(self.dataloader_large)} for MSE Init...", end='\r')
                lq = lq.to(self.device)
                # lq_center = get_center_crop(lq, patch_size=256)
                lq_patches = get_random_patches_with_aug(lq, patch_size=128)
                _ = self.net_Q(lq_patches) 
        print("\n  > MSE-based Initialization Completed.")

        for name, module in self.net_Q.named_modules():
             if hasattr(module, 'calibrated'): module.calibrated = True

        # [Snapshot] Save Initialized Bounds for later comparison
        initial_bounds = {}
        for name, module in self.net_Q.named_modules():
            if hasattr(module, 'lower_bound') and hasattr(module, 'upper_bound'):
                initial_bounds[name] = (
                    module.lower_bound.detach().clone().cpu(), 
                    module.upper_bound.detach().clone().cpu()
                )

        # -------------------------------------------------------
        # [Step 2] Calibrate Wavelet Normalization Statistics (NEW)
        # -------------------------------------------------------
        print("\n[Step 2] Calibrating Wavelet Normalization Statistics (Using small Batch)...")
        self.swt_norm.train() # 학습 모드로 전환하여 running_mean/var 업데이트
        
        with torch.no_grad():
            for i, (lq, _) in enumerate(self.dataloader_large):
                print(f"  > Processing Batch {i+1}/{len(self.dataloader_large)} for Wavelet Stats...", end='\r')
                lq = lq.to(self.device)
                lq_center = get_center_crop(lq, patch_size=512)
                
                output_F = self.net_F(lq_center)
                fp_swt = serialize_swt(self.swt(output_F))
                
                _ = self.swt_norm(fp_swt, update_stat=True)
                
        self.swt_norm.eval() 
        print(f"\n  > Wavelet Stats Calibrated.")
        print(f"    Mean: {self.swt_norm.running_mean.cpu().numpy()}")
        print(f"    Var : {self.swt_norm.running_var.cpu().numpy()}")

        # -------------------------------------------------------
        # [Step 3] Wavelet-guided Fine Optimization 
        # -------------------------------------------------------
        print("\n[Step 3] Starting Wavelet-guided Fine Optimization...")

        weight_params = []
        act_params = []
        weight_cnt = 0
        act_cnt = 0
        for name, p in self.net_Q.named_parameters():
            if 'lower_bound' not in name and 'upper_bound' not in name:
                p.requires_grad = False
                continue
            p.requires_grad = False 
            if 'weight_quantizer' in name: 
                weight_params.append(p)
                weight_cnt += 1
            elif 'act_quantizer' in name or 'q_quantizer' in name or 'k_quantizer' in name or 'v_quantizer' in name or 'attn_quantizer' in name: 
                act_params.append(p)
                act_cnt += 1
                
        print(f"  > Total Quantization Parameters: {len(weight_params) + len(act_params)} (Weights: {weight_cnt}, Activations: {act_cnt})")

        optimizer_w = torch.optim.Adam(weight_params, lr=1e-4, betas=(0.9, 0.99)) if weight_params else None
        optimizer_a = torch.optim.Adam(act_params, lr=1e-3, betas=(0.9, 0.99)) if act_params else None

        scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=self.total_weight_steps, eta_min=1e-8) if optimizer_w else None
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=self.total_act_steps, eta_min=1e-8) if optimizer_a else None
        
        for phase_name, steps in self.schedule:
            print(f"\n>>> Phase: {phase_name} (Steps: {steps})")
            
            if weight_params:
                for p in weight_params: p.requires_grad = (phase_name in ['weight', 'both'])
            if act_params:
                for p in act_params: p.requires_grad = (phase_name in ['act', 'both'])

            pbar = tqdm(range(steps), desc=f"Fine-tuning {phase_name}")
            iter_count = 0
            
            running_loss = 0.0
            running_ll = 0.0
            running_hh = 0.0
            running_high = 0.0
            data_iter = iter(self.dataloader_small)

            while iter_count < steps:
                try:
                    lq, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader_small)
                    lq, _ = next(data_iter)
                
                lq = lq.to(self.device)
                
                full_patches = get_random_patches_with_aug(lq, patch_size=128)

                if phase_name in ['weight', 'both'] and optimizer_w: optimizer_w.zero_grad()
                if phase_name in ['act', 'both'] and optimizer_a: optimizer_a.zero_grad()

                output_Q = self.net_Q(full_patches)
                with torch.no_grad(): 
                    output_F = self.net_F(full_patches)

                q_swt = serialize_swt(self.swt(output_Q))
                q_swt_norm = self.swt_norm(q_swt, update_stat=False)
                with torch.no_grad():
                    fp_swt = serialize_swt(self.swt(output_F))
                    fp_swt_norm = self.swt_norm(fp_swt, update_stat=False)

                q_ll = q_swt_norm[:, :1]
                fp_ll = fp_swt_norm[:, :1]
                l_ll = F.l1_loss(q_ll, fp_ll) 

                q_hh = q_swt_norm[:, [3, 6]]
                fp_hh = fp_swt_norm[:, [3, 6]]
                l_hh = F.l1_loss(q_hh, fp_hh)

                q_high = q_swt_norm[:, [1, 2, 4, 5]]
                fp_high = fp_swt_norm[:, [1, 2, 4, 5]]
                l_high = F.l1_loss(q_high, fp_high) 

                # l_total = l_ll + l_hh + l_high
                # l_total = 0.5 * l_ll + 0.8 * l_hh + 0.1 * l_high
                l_total = 0.2 * l_ll + 0.5 * l_hh + 0.3 * l_high

                l_total.backward()
                
                current_lrs = []
                if phase_name in ['weight', 'both'] and optimizer_w:
                    torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
                    optimizer_w.step()
                    scheduler_w.step()
                    current_lrs.append(f"W:{scheduler_w.get_last_lr()[0]:.1e}")

                if phase_name in ['act', 'both'] and optimizer_a:
                    torch.nn.utils.clip_grad_norm_(act_params, max_norm=1.0)
                    optimizer_a.step()
                    scheduler_a.step()
                    current_lrs.append(f"A:{scheduler_a.get_last_lr()[0]:.1e}")


                pbar.set_postfix({
                    'L': f"{l_total.item():.4f}", 
                    'LL': f"{l_ll.item():.4f}",
                    'HH': f"{l_hh.item():.4f}",
                    'High': f"{l_high.item():.4f}",
                    'LR': "|".join(current_lrs)
                })
                pbar.update(1)
                
                running_loss += l_total.item()
                running_ll += l_ll.item()
                running_hh += l_hh.item()
                running_high += l_high.item()

                iter_count += 1
                if iter_count % 100 == 0:
                    avg_loss = running_loss / 100
                    avg_hh = running_hh / 100
                    avg_ll = running_ll / 100
                    avg_high = running_high / 100
                    pbar.write(f"[{phase_name}] Iter {iter_count}: Avg L:{avg_loss:.5f} | LL: {avg_ll:.5f} | HH:{avg_hh:.5f} | High:{avg_high:.5f}")
                    running_loss, running_ll, running_hh, running_high = 0.0, 0.0, 0.0, 0.0

            pbar.close()


def quantize_model(original_model, bit, calibration_dataloader_small, calibration_dataloader_large, device, save_path):
    print(f"\nQuantizing to {bit}-bit using Ours (SWT Loss)...")

    fp_model = copy.deepcopy(original_model)
    fp_model = fp_model.to(device)
    fp_model.eval()

    model = original_model
    model = convert_to_quantized(model, bit)

    # -------------------------------------------------------------------
    # First and Last layers are always quantized to 8-bit
    # -------------------------------------------------------------------
    print(f"Applying Mixed Precision: Setting conv_first and conv_last to 8-bit...")
    
    # 1. conv_first
    if hasattr(model, 'conv_first'):
        if hasattr(model.conv_first, 'weight_quantizer'):
            model.conv_first.weight_quantizer.set_n_bit_manually(8)
            print("  > conv_first.weight_quantizer -> 8-bit")
        if hasattr(model.conv_first, 'act_quantizer'):
            model.conv_first.act_quantizer.set_n_bit_manually(8)
            print("  > conv_first.act_quantizer    -> 8-bit")
            
    # 2. conv_last
    if hasattr(model, 'conv_last'):
        if hasattr(model.conv_last, 'weight_quantizer'):
            model.conv_last.weight_quantizer.set_n_bit_manually(8)
            print("  > conv_last.weight_quantizer           -> 8-bit")
        if hasattr(model.conv_last, 'act_quantizer'):
            model.conv_last.act_quantizer.set_n_bit_manually(8)
            print("  > conv_last.act_quantizer            -> 8-bit")
            
    model = model.to(device)
    model.eval()

    if hasattr(model, 'model_init'):
         print("Running Initialization...")
         try:
            model.model_init(calibration_dataloader_large, device)
         except Exception as e:
            print(f"Init failed: {e}")
    
    optimizer_wrapper = OptimizationWrapper(model, fp_model, calibration_dataloader_small, calibration_dataloader_large, device)
    optimizer_wrapper.optimize()

    print(f"Saving to {save_path}")
    torch.save(model.state_dict(), save_path)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--train_gt', type=str, default='')
    parser.add_argument('--train_lq', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--bit', type=int, required=True, help="Bit width (8, 4, 2)")

    # SwinIR args
    parser.add_argument('--embed_dim', type=int, default=60)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size for SwinIR structure') # 128 for moving700, 256 for Mayo
    parser.add_argument('--batch_size', type=int, default=1, help='Optim batch size (generated from patches)')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"SwinIR_ours_{args.bit}bit.pth")

    # Dataset
    print("Initializing Calibration Dataset...")
    calib_dataset = CalibrationDataset(args.train_lq)
    calib_loader_small = DataLoader(calib_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    calib_loader_large = DataLoader(calib_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
     
    print(f"Loading SwinIR from {args.model_path}...")
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
        upsampler='',  
        resi_connection='1conv'
    )
    
    # Load pretrained weights
    pretrained_model = torch.load(args.model_path, map_location=device)
    param_key_g = 'params'
    if param_key_g in pretrained_model:
        model.load_state_dict(pretrained_model[param_key_g])
    else:
        model.load_state_dict(pretrained_model)
        
    model = model.to(device)
    quantize_model(model, args.bit, calib_loader_small, calib_loader_large, device, save_path)

if __name__ == '__main__':
    main()
