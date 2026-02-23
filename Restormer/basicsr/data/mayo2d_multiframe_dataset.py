from torch.utils import data as data
from torchvision.transforms.functional import normalize
from PIL import Image
import numpy as np
import torch
import os
from pathlib import Path

from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, img2tensor, padding


class Dataset_Mayo2D_MultiFrame(data.Dataset):
    def __init__(self, opt):
        super(Dataset_Mayo2D_MultiFrame, self).__init__()
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
        data_pairs = []
        
        lq_subfolders = sorted([d for d in self.lq_folder.iterdir() if d.is_dir()])
        
        for lq_subfolder in lq_subfolders:
            subfolder_name = lq_subfolder.name
            gt_subfolder = self.gt_folder / subfolder_name
            
            if not gt_subfolder.exists():
                print(f"Warning: GT subfolder {gt_subfolder} does not exist, skipping...")
                continue
            
            lq_files = sorted(lq_subfolder.glob("*.tiff")) + sorted(lq_subfolder.glob("*.tif"))
            gt_files = sorted(gt_subfolder.glob("*.tiff")) + sorted(gt_subfolder.glob("*.tif"))
            
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

    def _load_tiff(self, path):
        img = Image.open(path)
        img_array = np.array(img, dtype=np.float32)
        
        if img_array.max() > 1.0:
            img_array = img_array / img_array.max()
        
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        return img_array

    def __getitem__(self, index):
        """
        Returns: 
        lq (3,H,W)
        gt (1,H,W)
        and paths
        """
        index = index % len(self.frame_list)
        frame_info = self.frame_list[index]
        
        pair_idx = frame_info['pair_idx']
        frame_idx = frame_info['frame_idx']
        
        data_pair = self.data_pairs[pair_idx]
        
        # t-1, t, t+1 LQ frames (prev, curr, next)
        lq_prev = self._load_tiff(data_pair['lq_files'][frame_idx - 1])
        lq_curr = self._load_tiff(data_pair['lq_files'][frame_idx])
        lq_next = self._load_tiff(data_pair['lq_files'][frame_idx + 1])
        
        # GT 
        gt_curr = self._load_tiff(data_pair['gt_files'][frame_idx])
        
        # Stack LQ frames 
        img_lq = np.concatenate([lq_prev, lq_curr, lq_next], axis=2) 
        img_gt = gt_curr  
        
        # Paths 
        lq_path = str(data_pair['lq_files'][frame_idx])
        gt_path = str(data_pair['gt_files'][frame_idx])
        
        # Training augmentation
        if self.opt['phase'] == 'train':
            gt_size = self.opt.get('gt_size', 384)
            scale = self.opt.get('scale', 1)
            
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        
        return {
            'lq': img_lq,  
            'gt': img_gt, 
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.frame_list)
