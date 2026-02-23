import os
import random
import numpy as np
import torch
from torch.utils import data
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

class Mayo2dDataset(data.Dataset):
    def __init__(self, gt_root, lq_root, patch_size=512, is_train=True, n_patches=1):
        super(Mayo2dDataset, self).__init__()
        self.gt_root = Path(gt_root)
        self.lq_root = Path(lq_root)
        self.patch_size = patch_size
        self.is_train = is_train
        self.n_patches = n_patches
        
        self.data_pairs = self._collect_data_pairs()
        self.frame_list = self._build_frame_list()
        
    def _collect_data_pairs(self):
        data_pairs = []
        # Check if root contains files directly or subfolders
        lq_subfolders = sorted([d for d in self.lq_root.iterdir() if d.is_dir()])
        
        if not lq_subfolders:
             return []

        for lq_subfolder in lq_subfolders:
            subfolder_name = lq_subfolder.name
            gt_subfolder = self.gt_root / subfolder_name
            
            if not gt_subfolder.exists():
                continue
            
            lq_files = sorted(list(lq_subfolder.glob("*.tiff")) + list(lq_subfolder.glob("*.tif")))
            gt_files = sorted(list(gt_subfolder.glob("*.tiff")) + list(gt_subfolder.glob("*.tif")))
            
            if len(lq_files) < 3 or len(lq_files) != len(gt_files):
                continue
            
            data_pairs.append({
                'subfolder': subfolder_name,
                'lq_files': lq_files,
                'gt_files': gt_files
            })
        return data_pairs

    def _build_frame_list(self):
        frame_list = []
        for pair_idx, pair in enumerate(self.data_pairs):
            num_frames = len(pair['lq_files'])
            # Exclude first and last frame
            for frame_idx in range(1, num_frames - 1):
                frame_list.append({
                    'pair_idx': pair_idx,
                    'frame_idx': frame_idx
                })
        print(f"Dataset: Found {len(frame_list)} frames.")
        return frame_list

    def _load_img(self, path):
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        return img

    def _augment(self, img_lq, img_gt):
        # img_lq: (H, W, 3)
        # img_gt: (H, W, 1) or (H, W)
        
        # Random flip
        if random.random() < 0.5:
            img_lq = np.flip(img_lq, axis=1)
            img_gt = np.flip(img_gt, axis=1)
        if random.random() < 0.5:
            img_lq = np.flip(img_lq, axis=0)
            img_gt = np.flip(img_gt, axis=0)
            
        # Random rotate
        k = random.randint(0, 3)
        img_lq = np.rot90(img_lq, k)
        img_gt = np.rot90(img_gt, k)
        
        return img_lq.copy(), img_gt.copy()

    def __getitem__(self, index):
        frame_info = self.frame_list[index // self.n_patches]
        pair_idx = frame_info['pair_idx']
        frame_idx = frame_info['frame_idx']
        
        pair = self.data_pairs[pair_idx]
        
        # Load 3 frames (t-1, t, t+1)
        lq_0 = self._load_img(pair['lq_files'][frame_idx-1])
        lq_1 = self._load_img(pair['lq_files'][frame_idx])
        lq_2 = self._load_img(pair['lq_files'][frame_idx+1])
        
        gt = self._load_img(pair['gt_files'][frame_idx])
        
        # Stack LQ -> (H, W, 3)
        lq_stack = np.stack([lq_0, lq_1, lq_2], axis=-1)
        
        # GT -> (H, W, 1)
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=-1)
            
        H, W, _ = lq_stack.shape
        
        if self.is_train:
            # Random Crop
            if H > self.patch_size and W > self.patch_size: 
                rnd_h = random.randint(0, H - self.patch_size)
                rnd_w = random.randint(0, W - self.patch_size)
                
                lq_stack = lq_stack[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
                gt = gt[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
                
            # Augment
            lq_stack, gt = self._augment(lq_stack, gt)

        # To Tensor (C, H, W)
        lq_tensor = torch.from_numpy(np.ascontiguousarray(lq_stack.transpose(2, 0, 1))).float()
        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))).float()
        
        return lq_tensor, gt_tensor

    def __len__(self):
        return len(self.frame_list) * self.n_patches

class Moving700Dataset(data.Dataset):
    def __init__(self, gt_root, lq_root, patch_size=64, is_train=True, n_patches=4):
        super(Moving700Dataset, self).__init__()
        self.gt_root = Path(gt_root)
        self.lq_root = Path(lq_root)
        self.patch_size = patch_size
        self.is_train = is_train
        self.n_patches = n_patches
        
        self.data_pairs = self._collect_data_pairs()
        self.frame_list = self._build_frame_list()
        
    def _collect_data_pairs(self):
        data_pairs = []
        # Check if root contains files directly or subfolders
        lq_subfolders = sorted([d for d in self.lq_root.iterdir() if d.is_dir()])
        
        if not lq_subfolders:
             # Maybe flat structure? But Moving700 usually has subfolders
             return []

        for lq_subfolder in lq_subfolders:
            subfolder_name = lq_subfolder.name
            gt_subfolder = self.gt_root / subfolder_name
            
            if not gt_subfolder.exists():
                continue
            
            lq_files = sorted(list(lq_subfolder.glob("*.tiff")) + list(lq_subfolder.glob("*.tif")))
            gt_files = sorted(list(gt_subfolder.glob("*.tiff")) + list(gt_subfolder.glob("*.tif")))
            
            if len(lq_files) < 3 or len(lq_files) != len(gt_files):
                continue
            
            data_pairs.append({
                'subfolder': subfolder_name,
                'lq_files': lq_files,
                'gt_files': gt_files
            })
        return data_pairs

    def _build_frame_list(self):
        frame_list = []
        for pair_idx, pair in enumerate(self.data_pairs):
            num_frames = len(pair['lq_files'])
            # Exclude first and last frame
            for frame_idx in range(1, num_frames - 1):
                frame_list.append({
                    'pair_idx': pair_idx,
                    'frame_idx': frame_idx
                })
        print(f"Dataset: Found {len(frame_list)} frames.")
        return frame_list

    def _load_img(self, path):
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)
        return img

    def _augment(self, img_lq, img_gt):
        # img_lq: (H, W, 3)
        # img_gt: (H, W, 1) or (H, W)
        
        # Random flip
        if random.random() < 0.5:
            img_lq = np.flip(img_lq, axis=1)
            img_gt = np.flip(img_gt, axis=1)
        if random.random() < 0.5:
            img_lq = np.flip(img_lq, axis=0)
            img_gt = np.flip(img_gt, axis=0)
            
        # Random rotate
        k = random.randint(0, 3)
        img_lq = np.rot90(img_lq, k)
        img_gt = np.rot90(img_gt, k)
        
        return img_lq.copy(), img_gt.copy()

    def __getitem__(self, index):
        frame_info = self.frame_list[index // self.n_patches]
        pair_idx = frame_info['pair_idx']
        frame_idx = frame_info['frame_idx']
        
        pair = self.data_pairs[pair_idx]
        
        # Load 3 frames (t-1, t, t+1)
        lq_0 = self._load_img(pair['lq_files'][frame_idx-1])
        lq_1 = self._load_img(pair['lq_files'][frame_idx])
        lq_2 = self._load_img(pair['lq_files'][frame_idx+1])
        
        gt = self._load_img(pair['gt_files'][frame_idx])
        
        # Stack LQ -> (H, W, 3)
        lq_stack = np.stack([lq_0, lq_1, lq_2], axis=-1)
        
        # GT -> (H, W, 1)
        if gt.ndim == 2:
            gt = np.expand_dims(gt, axis=-1)
            
        H, W, _ = lq_stack.shape
        
        if self.is_train:
            # Random Crop
            if H > self.patch_size and W > self.patch_size: # 700x700 guaranteed, but check
                rnd_h = random.randint(0, H - self.patch_size)
                rnd_w = random.randint(0, W - self.patch_size)
                
                lq_stack = lq_stack[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
                gt = gt[rnd_h:rnd_h+self.patch_size, rnd_w:rnd_w+self.patch_size, :]
                
            # Augment
            lq_stack, gt = self._augment(lq_stack, gt)

        # To Tensor (C, H, W)
        lq_tensor = torch.from_numpy(np.ascontiguousarray(lq_stack.transpose(2, 0, 1))).float()
        gt_tensor = torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))).float()
        
        return lq_tensor, gt_tensor

    def __len__(self):
        return len(self.frame_list) * self.n_patches
