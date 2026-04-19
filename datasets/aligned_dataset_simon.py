import glob
import os.path
import random
import numpy as np
import pandas as pd 
import torchio as tio 
import torch
import torchvision.transforms as transforms
from PIL import Image
from datasets.base_dataset import BaseDataset
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from datasets.select_mask import define_Mask
from scipy.io import *
from scipy.fftpack import *

class AlignedDatasetSIMON(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.all_data = []
        
        self.df = pd.read_csv(self.root)
        self.df = self.df.loc[self.df['fold'] == opt.phase]
        
        self.mask = define_Mask(self.opt)
        self.mask = torch.from_numpy(self.mask).unsqueeze(0)
        self.mask = self.roll(self.mask, 128, 1)
        self.mask = self.roll(self.mask, 128, 2)
        self.mask_return = self.mask.permute(1, 2, 0)
        
        self._preload_all_data()
        
        print(f"---------------------------{opt.phase} images: {len(self.all_data)} ---------------------------")

    def _preload_all_data(self):
        source_str = '/home/sxb/Home_Folder_File/SMI_PRO_ST'
        fix_str = '/media/potato/cyf/cyf'
        
        file_groups = {}
        for idx in range(len(self.df)):
            _, slice_indice,  _, t2_volumn_path, pd_volumn_path, flair_volumn_path, t2star, _ = self.df.iloc[idx].tolist()
            
            t2_path = t2_volumn_path.replace(source_str, fix_str)
            pd_path = pd_volumn_path.replace(source_str, fix_str)
            flair_path = flair_volumn_path.replace(source_str, fix_str)
            t2star_path = t2star.replace(source_str, fix_str)
            
            if slice_indice >= 2:
                slice_n1 = slice_indice - 1
            else:
                slice_n1 = slice_indice + 1
            
            group_key = t2_path
            if group_key not in file_groups:
                file_groups[group_key] = {
                    'pd_path': pd_path,
                    'flair_path': flair_path,
                    't2star_path': t2star_path,
                    'slices': []
                }
            file_groups[group_key]['slices'].append({
                'original_idx': idx,
                'slice_indice': slice_indice,
                'slice_n1': slice_n1
            })
        
        pbar = tqdm(file_groups.items(), desc=f"Preloading {self.opt.phase} volumes")
        for group_key, group_info in pbar:
            try:
                t2_volume = tio.ScalarImage(group_key).data  # shape: [1, H, W, D]
                pd_volume = tio.ScalarImage(group_info['pd_path']).data
                
                for slice_info in group_info['slices']:
                    slice_indice = slice_info['slice_indice']
                    slice_n1 = slice_info['slice_n1']
                    
                    t2_data = t2_volume[..., slice_n1]
                    pd_data = pd_volume[..., slice_n1]
                    
                    if self.opt.phase == 'train':
                        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
                        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
                    else:
                        w_offset = (self.opt.loadSize - self.opt.fineSize) // 2
                        h_offset = (self.opt.loadSize - self.opt.fineSize) // 2
                    
                    t2_data = self._normalize_and_crop(t2_data, h_offset, w_offset)
                    pd_data = self._normalize_and_crop(pd_data, h_offset, w_offset)
                    
                    an1_down_k, an1_down_img = self.undersample_kspace(t2_data, self.mask)
                    an1_full_k, an1_full_img = self.undersample_kspace_fake(t2_data, self.mask)
                    
                    bn1_down_k, bn1_down_img = self.undersample_kspace(pd_data, self.mask)
                    bn1_full_k, bn1_full_img = self.undersample_kspace_fake(pd_data, self.mask)
                    
                    data_item = {
                        'ref_kspace_fullN1': bn1_full_k,
                        'ref_kspace_subN1': bn1_down_k,
                        'ref_image_fullN1': bn1_full_img,
                        'ref_image_subN1': bn1_down_img,
                        'ref_kspace_mask2d': self.mask_return,
                        
                        'tag_kspace_fullN1': an1_full_k,
                        'tag_kspace_subN1': an1_down_k,
                        'tag_image_fullN1': an1_full_img,
                        'tag_image_subN1': an1_down_img,
                        'tag_kspace_mask2d': self.mask_return,
                        
                        'A_paths': group_key,
                        'B_paths': group_info['pd_path'],
                        'C_paths': group_info['flair_path'],
                        'D_paths': group_info['t2star_path'],
                        'slice_idx': int(slice_indice)
                    }
                    
                    self.all_data.append(data_item)
                
                del t2_volume, pd_volume
                
            except Exception as e:
                pbar.set_postfix({'error': str(e)[:50]})
                print(f"\nError loading volume {group_key}: {e}")
                continue
        
        pbar.close()

    def _normalize_and_crop(self, data, h_offset, w_offset):
        data = (data - data.min()) / (data.max() - data.min() + 1e-8) 
        data = data[:, h_offset:h_offset + self.opt.fineSize, 
                    w_offset:w_offset + self.opt.fineSize]
        return data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def name(self):
        return 'AlignedDatasetSIMON'

    def roll(self, tensor, shift, axis):
        if shift == 0:
            return tensor
        if axis < 0:
            axis += tensor.dim()
        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)
        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)

    def undersample_kspace(self, x, mask):
        x_k = torch.fft.fft2(x, dim=(-2, -1))
        x_k1 = torch.stack((x_k.real, x_k.imag), dim=-1)
        x_k1_real = x_k1[..., 0]
        x_k1_imag = x_k1[..., 1]
        x_k_return = torch.cat((x_k1_real, x_k1_imag), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        x_k_return = x_k_return * mask
        x_k1_return_real = x_k_return[0, ...].unsqueeze(0)
        x_k1_return_imag = x_k_return[1, ...].unsqueeze(0)
        x_k2 = torch.complex(x_k1_return_real, x_k1_return_imag)
        x_image = torch.fft.ifft2(x_k2, dim=(-2, -1))
        x_image_return = torch.cat((x_image.real, x_image.imag), 0)
        return x_k_return.permute(1, 2, 0), x_image_return

    def undersample_kspace_fake(self, x, mask):
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft1 = torch.cat((x_fft.real, x_fft.imag), 0)
        x_k_full_return = x_fft1
        x_image_full = torch.fft.ifft2(x_fft, dim=(-2, -1))
        x_image_full_return = torch.cat((x_image_full.real, x_image_full.imag), 0)
        return x_k_full_return.permute(1, 2, 0), x_image_full_return