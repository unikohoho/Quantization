from torch.utils import data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding


class Dataset_Moving700_MultiFrame(data.Dataset):
    """
    Multi-frame dataset for Moving700
    - Input: 3 consecutive frames stacked (t-1, t, t+1)
    - Output: Center frame (t)
    """
    def __init__(self, opt):
        super(Dataset_Moving700_MultiFrame, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        self.gt_folder = Path(opt['dataroot_gt'])
        self.lq_folder = Path(opt['dataroot_lq'])
        
        self.data_pairs = self._collect_data_pairs()
        
        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', False)

    def _collect_data_pairs(self):
        """
        Moving700 structure:
        - train/high/static_set1/static_set1-000.png, ...
        - train/low/static_set1/static_set1-000.png, ...
        Multiple subfolders: static_set1, rotate_set1, hand_linear_set1, etc.
        """
        data_pairs = []
        
        lq_subfolders = sorted([d for d in self.lq_folder.iterdir() if d.is_dir()])
        
        for lq_subfolder in lq_subfolders:
            subfolder_name = lq_subfolder.name
            gt_subfolder = self.gt_folder / subfolder_name
            
            if not gt_subfolder.exists():
                print(f"Warning: GT subfolder {gt_subfolder} does not exist, skipping...")
                continue
            
            lq_files = sorted(lq_subfolder.glob("*.png")) + sorted(lq_subfolder.glob("*.tiff")) + sorted(lq_subfolder.glob("*.tif"))
            gt_files = sorted(gt_subfolder.glob("*.png")) + sorted(gt_subfolder.glob("*.tiff")) + sorted(gt_subfolder.glob("*.tif"))
            
            if len(lq_files) != len(gt_files):
                print(f"Warning: Mismatch in file count for {subfolder_name}: "
                      f"LQ={len(lq_files)}, GT={len(gt_files)}, skipping...")
                continue
            
            if len(lq_files) < 3:
                print(f"Warning: Not enough frames in {subfolder_name} (need at least 3), skipping...")
                continue
            
            data_pairs.append({
                'subfolder': subfolder_name,
                'lq_files': lq_files,
                'gt_files': gt_files
            })
        
        # Build frame list (exclude first and last frame of each subfolder) -> 
        self.frame_list = []
        for pair_idx, pair in enumerate(data_pairs):
            num_frames = len(pair['lq_files'])
            for frame_idx in range(1, num_frames - 1):
                self.frame_list.append({
                    'pair_idx': pair_idx,
                    'frame_idx': frame_idx
                })
        
        print(f"Collected {len(self.frame_list)} valid frames from {len(data_pairs)} subfolders")
        
        return data_pairs

    def _load_image(self, path):
        """Load image using basicsr's imfrombytes (handles TIFF 32-bit correctly)"""
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt['type'])
        
        img_bytes = self.file_client.get(str(path))
        img = imfrombytes(img_bytes, float32=True)  # Returns [0, 1] range, float32
        
        # Convert grayscale to (H, W, 1) format
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        
        return img

    def __getitem__(self, index):
        """
        Returns: 
        lq: (3, H, W) - 3 consecutive LQ frames stacked
        gt: (1, H, W) - Center frame GT
        """
        index = index % len(self.frame_list)
        frame_info = self.frame_list[index]
        
        pair_idx = frame_info['pair_idx']
        frame_idx = frame_info['frame_idx']
        
        data_pair = self.data_pairs[pair_idx]
        
        # Load t-1, t, t+1 LQ frames (prev, curr, next)
        lq_prev = self._load_image(data_pair['lq_files'][frame_idx - 1])
        lq_curr = self._load_image(data_pair['lq_files'][frame_idx])
        lq_next = self._load_image(data_pair['lq_files'][frame_idx + 1])
        
        # Load GT for center frame
        gt_curr = self._load_image(data_pair['gt_files'][frame_idx])
        
        # Stack LQ frames along channel dimension
        img_lq = np.concatenate([lq_prev, lq_curr, lq_next], axis=2)  # (H, W, 3)
        img_gt = gt_curr  # (H, W, 1)
        
        # Paths
        lq_path = str(data_pair['lq_files'][frame_idx])
        gt_path = str(data_pair['gt_files'][frame_idx])
        
        # Training augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size', 384)
            scale = self.opt.get('scale', 1)
            
            # Padding if needed
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            
            # Random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            # Geometric augmentation (flip, rotate)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        
        # Convert to tensor
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        
        # No normalization - keep data in [0, 1] range
        
        return {
            'lq': img_lq,  # (3, H, W)
            'gt': img_gt,  # (1, H, W)
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.frame_list)
