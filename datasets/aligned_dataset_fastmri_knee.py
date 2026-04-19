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
import h5py

from datasets.select_mask import define_Mask
from scipy.io import *
from scipy.fftpack import *

### fastMRI
class AlignedDataset2(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.df = pd.read_csv(self.root)
        self.df = self.df.loc[self.df['fold'] == opt.phase]
        print("DataFrame columns:", list(self.df.columns))
        
        self.cache = {}   # key: (v_idx, s_idx) -> dict(pd=..., pdfs=...)
        self._preload_all_slices()

        self.mask = define_Mask(self.opt)

        print(f"---------------------------{opt.phase} images: {len(self.df)} ---------------------------")


    def _preload_all_slices(self):
        grouped = self.df.groupby(['volumn_indices', 'pd_volumn_paths', 'pdfs_volumn_paths'])
        for (v_idx, pd_path, pdfs_path), grp in grouped:
            with h5py.File(pd_path, 'r') as f_pd, \
                h5py.File(pdfs_path, 'r') as f_pdfs:
                pd_vol   = f_pd['reconstruction_rss'][:]   # (S, H, W)
                pdfs_vol = f_pdfs['reconstruction_rss'][:]

            # 直接转成 float32 tensor，省得后面再 from_numpy
            self.cache[int(v_idx)] = {
                'pd':   torch.from_numpy(pd_vol.astype(np.float32)),
                'pdfs': torch.from_numpy(pdfs_vol.astype(np.float32))
            }


    def __getitem__(self, index):
        row = self.df.iloc[index]
        v_idx = int(row['volumn_indices'])
        s_idx = int(row['slice_indices'])

        # 整卷 tensor，[S, H, W]
        pd_vol   = self.cache[v_idx]['pd']
        pdfs_vol = self.cache[v_idx]['pdfs']

        A_paths = row['pd_volumn_paths']
        B_paths = row['pdfs_volumn_paths']

        # 3 张连续切片，一次性切片，0 拷贝
        if s_idx >= 2:
            A   = pd_vol[s_idx].unsqueeze(0)      # [1, H, W]
            AN1 = pd_vol[s_idx-1].unsqueeze(0)
            AN2 = pd_vol[s_idx-2].unsqueeze(0)

            B   = pdfs_vol[s_idx].unsqueeze(0)
            BN1 = pdfs_vol[s_idx-1].unsqueeze(0)
            BN2 = pdfs_vol[s_idx-2].unsqueeze(0)
        else:  # 用 +2, +1, 0 顺序
            A   = pd_vol[s_idx+2].unsqueeze(0)
            AN1 = pd_vol[s_idx+1].unsqueeze(0)
            AN2 = pd_vol[s_idx].unsqueeze(0)

            B   = pdfs_vol[s_idx+2].unsqueeze(0)
            BN1 = pdfs_vol[s_idx+1].unsqueeze(0)
            BN2 = pdfs_vol[s_idx].unsqueeze(0)


        # mask
        mask = self.mask

        # 将mask数组扩展一个维度
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        mask = self.roll(mask, 160, 1)
        mask = self.roll(mask, 160, 2)


        # 数据归一化
        A = (A - A.min()) / (A.max() - A.min())

        # 数据裁剪
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        # 返回欠采样k空间数据，欠采样图像
        A_downsample_k, A_downsample_image = self.undersample_kspace(A, mask)
        # 返回全采样k空间数据，全采样图像
        A_full_k, A_full_image = self.undersample_kspace_fake(A, mask)

        ##############################################################################################################################

        AN1 = (AN1 - AN1.min()) / (AN1.max() - AN1.min())
        AN1 = AN1[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        AN1_downsample_k, AN1_downsample_image = self.undersample_kspace(AN1, mask)
        AN1_full_k, AN1_full_image = self.undersample_kspace_fake(AN1, mask)

        ##############################################################################################################################

        AN2 = (AN2 - AN2.min()) / (AN2.max() - AN2.min())
        AN2 = AN2[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        AN2_downsample_k, AN2_downsample_image = self.undersample_kspace(AN2, mask)
        AN2_full_k, AN2_full_image = self.undersample_kspace_fake(AN2, mask)

        ##############################################################################################################################
        ##############################################################################################################################
        
        B = (B - B.min()) / (B.max() - B.min())
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B_downsample_k, B_downsample_image = self.undersample_kspace(B, mask)
        B_full_k, B_full_image = self.undersample_kspace_fake(B, mask)

        ##############################################################################################################################

        BN1 = (BN1 - BN1.min()) / (BN1.max() - BN1.min())
        BN1 = BN1[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        BN1_downsample_k, BN1_downsample_image = self.undersample_kspace(BN1, mask)
        BN1_full_k, BN1_full_image = self.undersample_kspace_fake(BN1, mask)

        ##############################################################################################################################

        BN2 = (BN2 - BN2.min()) / (BN2.max() - BN2.min())
        BN2 = BN2[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        BN2_downsample_k, BN2_downsample_image = self.undersample_kspace(BN2, mask)
        BN2_full_k, BN2_full_image = self.undersample_kspace_fake(BN2, mask)

        mask_return = mask.permute(1, 2, 0)

        return {'ref_kspace_full': B_full_k, 'ref_kspace_sub': B_downsample_k, 'ref_image_full': B_full_image, 'ref_image_sub': B_downsample_image,
                'ref_kspace_fullN1': BN1_full_k, 'ref_kspace_subN1': BN1_downsample_k, 'ref_image_fullN1': BN1_full_image, 'ref_image_subN1': BN1_downsample_image,
                'ref_kspace_fullN2': BN2_full_k, 'ref_kspace_subN2': BN2_downsample_k, 'ref_image_fullN2': BN2_full_image, 'ref_image_subN2': BN2_downsample_image,
                'ref_kspace_mask2d': mask_return,
                #
                'tag_kspace_full': A_full_k, 'tag_kspace_sub': A_downsample_k, 'tag_image_full': A_full_image, 'tag_image_sub': A_downsample_image,
                'tag_kspace_fullN1': AN1_full_k, 'tag_kspace_subN1': AN1_downsample_k, 'tag_image_fullN1': AN1_full_image, 'tag_image_subN1': AN1_downsample_image,
                'tag_kspace_fullN2': AN2_full_k, 'tag_kspace_subN2': AN2_downsample_k, 'tag_image_fullN2': AN2_full_image, 'tag_image_subN2': AN2_downsample_image,
                'tag_kspace_mask2d': mask_return,
                #
                'flair_kspace_full': A_full_k, 'flair_kspace_sub': A_downsample_k, 'flair_image_full': A_full_image, 'flair_image_sub': A_downsample_image,
                'flair_kspace_fullN1': AN1_full_k, 'flair_kspace_subN1': AN1_downsample_k, 'flair_image_fullN1': AN1_full_image, 'flair_image_subN1': AN1_downsample_image,
                'flair_kspace_fullN2': AN2_full_k, 'flair_kspace_subN2': AN2_downsample_k, 'flair_image_fullN2': AN2_full_image, 'flair_image_subN2': AN2_downsample_image,
                'flair_kspace_mask2d': mask_return,
                #
                't2star_kspace_full': A_full_k, 't2star_kspace_sub': A_downsample_k, 't2star_image_full': A_full_image, 't2star_image_sub': A_downsample_image,
                't2star_kspace_fullN1': AN1_full_k, 't2star_kspace_subN1': AN1_downsample_k, 't2star_image_fullN1': AN1_full_image, 't2star_image_subN1': AN1_downsample_image,
                't2star_kspace_fullN2': AN2_full_k, 't2star_kspace_subN2': AN2_downsample_k, 't2star_image_fullN2': AN2_full_image, 't2star_image_subN2': AN2_downsample_image,
                't2star_kspace_mask2d': mask_return,
                #
                'A_paths': A_paths, 'B_paths': B_paths, 'C_paths': A_paths, 'D_paths': A_paths, 
                #
                'slice_idx': int(s_idx)}



    def __len__(self):
        return (len(self.df))

    def name(self):
        return 'AlignedDataset2'

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
        # print(x.shape)
        # print(mask.shape)
        x_k = torch.fft.fft2(x,dim=(-2,-1))
        # print(x_k.shape)
        x_k1 = torch.stack((x_k.real, x_k.imag), dim=-1)
        x_k1_real = x_k1[...,0]
        x_k1_imag = x_k1[...,1]
        x_k_return = torch.cat((x_k1_real, x_k1_imag), dim=0)
        mask = torch.cat((mask, mask), dim=0)
        x_k_return = x_k_return * mask
        x_k1_return_real = x_k_return[0,...].unsqueeze(0)
        x_k1_return_imag = x_k_return[1,...].unsqueeze(0)
        x_k2 = torch.complex(x_k1_return_real, x_k1_return_imag)
        x_image = torch.fft.ifft2(x_k2, dim=(-2, -1))
        x_image_return = torch.cat((x_image.real,x_image.imag),0)
        # print(x_k_return.shape)
        # print(x_image_return.shape)
        return x_k_return.permute(1, 2, 0), x_image_return


    def undersample_kspace_fake(self, x, mask):
        x_fft = torch.fft.fft2(x,dim=(-2,-1))
        x_fft1 = torch.cat((x_fft.real,x_fft.imag),0)
        x_k_full_return = x_fft1
        x_image_full = torch.fft.ifft2(x_fft , dim=(-2, -1)) 
        x_image_full_return = torch.cat((x_image_full.real,x_image_full.imag),0)
        # print(x_k_full_return.shape)
        return x_k_full_return.permute(1, 2, 0), x_image_full_return



