import os
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from networks import get_generator
from networks.networks import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, get_nonlinearity, DataConsistencyInKspace_I, DataConsistencyInKspace_K, fft2, complex_abs_eval

import pdb

import numpy as np
import lpips
import utils.utils_image as util
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2


import torch
import torch.nn as nn
from pytorch_msssim import ssim as ssim_pytorch

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.01, data_range=1.0):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.data_range = data_range
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def sigtoimage(self, sig):
        x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
        x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
        x_image = torch.sqrt(x_real * x_real + x_imag * x_imag + 1e-8)
        return x_image

    def forward(self, pred, gt):
        loss_pixel = self.l1_loss(pred, gt)
        
        loss_mse = self.mse_loss(pred, gt)
        
        pred_mag = self.sigtoimage(pred)
        gt_mag = self.sigtoimage(gt)
        
        ssim_val = ssim_pytorch(pred_mag, gt_mag, data_range=self.data_range, size_average=True)
        loss_ssim = 1.0 - ssim_val
        
        # 4. 加权求和总损失
        loss_total = (self.alpha * loss_pixel) + (self.beta * loss_ssim) + (self.gamma * loss_mse)
        
        return loss_total



class RecurrentModel(nn.Module):
    def __init__(self, opts):
        super(RecurrentModel, self).__init__()

        self.loss_names = []
        self.networks = []
        self.optimizers = []

        self.n_recurrent = opts.n_recurrent

        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G_I = get_generator(opts.net_G, opts)
        # self.net_G_K = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G_I)
        # self.networks.append(self.net_G_K)

        if self.is_train:
            self.loss_names += ['loss_G_L1']
            # param = list(self.net_G_I.parameters()) + list(self.net_G_K.parameters())
            param = list(self.net_G_I.parameters())
            self.optimizer_G = torch.optim.Adam(param,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.L1Loss()
        # self.criterion = HybridLoss()

        self.opts = opts

        # data consistency layers in image space & k-space
        dcs_I = []
        for i in range(self.n_recurrent):
            dcs_I.append(DataConsistencyInKspace_I(noise_lvl=None))
        self.dcs_I = dcs_I

        # dcs_K = []
        # for i in range(self.n_recurrent):
        #     dcs_K.append(DataConsistencyInKspace_K(noise_lvl=None))
        # self.dcs_K = dcs_K

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.pd_kspace_mask2d = data['ref_kspace_mask2d'].to(self.device).float()

        self.pd_kspace_full = data['ref_kspace_fullN1'].to(self.device).float()
        self.pd_kspace_sub = data['ref_kspace_subN1'].to(self.device).float()
        self.pd_image_full = data['ref_image_fullN1'].to(self.device).float()
        self.pd_image_sub = data['ref_image_subN1'].to(self.device).float()

        self.t2_kspace_mask2d = data['tag_kspace_mask2d'].to(self.device).float()

        self.t2_kspace_full = data['tag_kspace_fullN1'].to(self.device).float()
        self.t2_kspace_sub = data['tag_kspace_subN1'].to(self.device).float()
        self.t2_image_full = data['tag_image_fullN1'].to(self.device).float()
        self.t2_image_sub = data['tag_image_subN1'].to(self.device).float()

        self.image_paths = data['A_paths']
        self.slice_idx = data['slice_idx']


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_slice_idx(self):
        return self.slice_idx

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        I = self.t2_image_sub
        I.requires_grad_(True)

        net = {}


        for i in range(1, self.n_recurrent + 1):
            '''Image Space'''
            if self.opts.use_prior == 2:
                x_I = torch.cat((I, self.pd_image_sub), 1)

            else:
                x_I = I

            net['r%d_img_pred' % i] = self.net_G_I(x_I)  # output recon image
            
            net['r%d_img_dc_pred' % i], _ = self.dcs_I[i - 1](net['r%d_img_pred' % i], self.t2_kspace_full, self.t2_kspace_mask2d)

            I = net['r%d_img_dc_pred' % i]


            
        self.net = net
        self.recon = I


    def update_G(self):
        loss_G_L1 = 0
        self.optimizer_G.zero_grad()

        # Image domain
        loss_img_dc = 0
        for j in range(1, self.n_recurrent + 1):
            loss_img_dc = loss_img_dc + self.criterion(self.net['r%d_img_dc_pred' % j], self.t2_image_full)

        loss_G_L1 = loss_img_dc
        self.loss_G_L1 = loss_G_L1.item()
        self.loss_img = loss_img_dc.item()

        total_loss = loss_G_L1
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_L1 = 0

        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        if self.opts.wr_L1 > 0:
            # message += 'G_L1: {:.4e} Img_L1: {:.4e} Kspc_L1: {:.4e}'.format(self.loss_G_L1, self.loss_img, self.loss_kspc)
            message += 'G_L1: {:.4e} Img_L1: {:.4e}'.format(self.loss_G_L1, self.loss_img)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        if self.opts.wr_L1 > 0:
            state['net_G_I'] = self.net_G_I.module.state_dict()
            # state['net_G_K'] = self.net_G_K.module.state_dict()
            state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        if self.opts.wr_L1 > 0:
            self.net_G_I.module.load_state_dict(checkpoint['net_G_I'], strict=False)
            # self.net_G_K.module.load_state_dict(checkpoint['net_G_K'])
            if train:
                self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    # Two channels image to magnitude image
    def sigtoimage(self, sig):
        x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
        x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
        x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
        return x_image

    def print_log(self, logger,message):
        print(message, flush=True)
        if logger:
            logger.write(str(message) + '\n')

    def evaluate(self, loader, epoch):        

        mae_avg = np.zeros([self.opts.how_many_valid])
        psnr_avg = np.zeros([self.opts.how_many_valid])
        ssim_avg = np.zeros([self.opts.how_many_valid])

        epoch_current = epoch

        output_directory = os.path.join(self.opts.output_path, self.opts.experiment_name)
        logger = open(os.path.join(output_directory, 'log.txt'), 'a')

        for i, data in enumerate(loader):
            
            if i >= self.opts.how_many_valid:
                break

            self.set_input(data)
            self.forward()

        
            self.pd_image_sub_onechannel = self.sigtoimage(self.pd_image_sub)
            self.t2_image_full_onechannel = self.sigtoimage(self.t2_image_full)
            self.t2_image_sub_onechannel = self.sigtoimage(self.t2_image_sub)
            self.recon_onechannel = self.sigtoimage(self.recon)

            fake_im_ = self.recon_onechannel.squeeze(0).cpu().data.numpy()
            real_im_ = self.t2_image_full_onechannel.squeeze(0).cpu().data.numpy()
            ZF_im = self.t2_image_sub_onechannel.squeeze(0).cpu().data.numpy()

            mae_avg[i] = abs(fake_im_-real_im_).mean()
            psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
            ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, channel_axis=2)

        mae = np.mean(mae_avg)
        mae_std = np.std(mae_avg)
        mean_psnr = np.mean(psnr_avg)
        std_psnr = np.std(psnr_avg)
        mean_ssim = np.mean(ssim_avg)
        std_ssim = np.std(ssim_avg)
        
        self.print_log(logger, '[Epoch %3d] psnr_mean: %.3f, psnr_std:%.3f, ssim_mean: %.3f, ssim_std:%.3f' % (epoch_current, mean_psnr, std_psnr, mean_ssim, std_ssim))
        self.print_log(logger, '')
        logger.close()

        return mean_psnr, mean_ssim


    def test(self, loader):
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()

        recon_images = []
        gt_images = []
        input_images = []

        mae_avg = np.zeros([self.opts.how_many])
        psnr_avg = np.zeros([self.opts.how_many])
        ssim_avg = np.zeros([self.opts.how_many])
        lpips_avg = np.zeros([self.opts.how_many])

        another_ssim_avg = np.zeros([self.opts.how_many])
        lips_rec_avg = np.zeros([self.opts.how_many])

        zf_mae_avg = np.zeros([self.opts.how_many])
        zf_psnr_avg = np.zeros([self.opts.how_many])
        zf_ssim_avg = np.zeros([self.opts.how_many])
        zf_lpips_avg = np.zeros([self.opts.how_many])

        another_zf_ssim_avg = np.zeros([self.opts.how_many])
        lips_zf_avg = np.zeros([self.opts.how_many])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)


        total_time = 0.0
        import time

        for i, data in enumerate(loader):
            
            if i >= self.opts.how_many:
                break

            self.set_input(data)
            start = time.perf_counter()
            self.forward()
            end = time.perf_counter()
            execution_time = (end - start)
            total_time += execution_time

            self.pd_image_sub_onechannel = self.sigtoimage(self.pd_image_sub)
            self.t2_image_full_onechannel = self.sigtoimage(self.t2_image_full)
            self.t2_image_sub_onechannel = self.sigtoimage(self.t2_image_sub)
            self.recon_onechannel = self.sigtoimage(self.recon)


            lpips_reconstruct_im = self.recon_onechannel
            lpips_grountruth_im = self.t2_image_full_onechannel
            lpips_ZF_im = self.t2_image_sub_onechannel

            # evaluate lpips
            lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_grountruth_im, lpips_reconstruct_im)
            lpips_ = lpips_.data.squeeze().float().cpu().numpy()
            lips_rec_avg[i] = lpips_
            # evaluate lpips zf
            zf_lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_grountruth_im, lpips_ZF_im)
            zf_lpips_ = zf_lpips_.data.squeeze().float().cpu().numpy()
            lips_zf_avg[i] = zf_lpips_

            fake_im_ = self.recon_onechannel.squeeze(0).cpu().data.numpy()
            real_im_ = self.t2_image_full_onechannel.squeeze(0).cpu().data.numpy()
            ZF_im = self.t2_image_sub_onechannel.squeeze(0).cpu().data.numpy()

            ssim_fake_im_ = fake_im_.squeeze(0)
            ssim_real_im_ = real_im_.squeeze(0)
            ssim_ZF_im = ZF_im.squeeze(0)

            another_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_fake_im_)
            another_zf_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_ZF_im)

            mae_avg[i] = abs(fake_im_-real_im_).mean()
            psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
            ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, channel_axis=2)


            zf_mae_avg[i] = abs(ZF_im-real_im_).mean()
            zf_psnr_avg[i] = psnr(real_im_, ZF_im, data_range=1)
            zf_ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(ZF_im, (1, 2, 0)), data_range=1, channel_axis=2)


            real_assistant = self.pd_image_sub_onechannel.squeeze(0).cpu().data.numpy()
            input_im_ = self.t2_image_sub_onechannel.squeeze(0).cpu().data.numpy()
            img_path = self.get_image_paths()
            slice_idx = self.get_slice_idx()
            volumn_name = img_path[0].split('/')[-1].split('.')[0]
            print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx))


            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'ZF'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'ZF'))
            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'GT'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'GT'))
            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'Recon'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'Recon'))
            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'Different_zero'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'Different_zero'))
            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'Different_rec'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'Different_rec'))
            isExists = os.path.exists(os.path.join(self.opts.results_dir, 'Assist'))
            if not isExists:
                os.makedirs(os.path.join(self.opts.results_dir, 'Assist'))



            fake_im_for_save = fake_im_.squeeze(0)
            real_im_for_save = real_im_.squeeze(0)
            real_assistant_for_save = real_assistant.squeeze(0)
            input_im_for_save = input_im_.squeeze(0)
            fake_image_for_df = self.recon_onechannel
            real_image_for_df = self.t2_image_full_onechannel
            input_image_for_df = self.t2_image_sub_onechannel

            diff_gen_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, fake_image_for_df)), 5)
            diff_lq_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, input_image_for_df)), 5)

            input_im_for_save = (np.clip(input_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
            fake_im_for_save = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
            real_im_for_save = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
            real_assistant_for_save = (np.clip(real_assistant_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

            diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

            diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
            diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8

            diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
            diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)

            input_image_dir = os.path.join(self.opts.results_dir, 'ZF' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
            fake_image_dir = os.path.join(self.opts.results_dir, 'Recon' + "/"+ str(volumn_name) + str(slice_idx.numpy()) + ".png")
            real_image_dir = os.path.join(self.opts.results_dir, 'GT' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
            assistant_image_dir = os.path.join(self.opts.results_dir, 'Assist' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
            diff_gen_image_dir = os.path.join(self.opts.results_dir, 'Different_rec' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
            diff_lq_image_dir = os.path.join(self.opts.results_dir, 'Different_zero' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")

            cv2.imwrite(input_image_dir, input_im_for_save)
            cv2.imwrite(fake_image_dir, fake_im_for_save)
            cv2.imwrite(real_image_dir, real_im_for_save)
            cv2.imwrite(assistant_image_dir, real_assistant_for_save)
            cv2.imwrite(diff_gen_image_dir, diff_gen_x10_color)
            cv2.imwrite(diff_lq_image_dir, diff_lq_x10_color)


        mae = np.mean(mae_avg)
        mae_std = np.std(mae_avg)
        mean_psnr = np.mean(psnr_avg)
        std_psnr = np.std(psnr_avg)
        mean_ssim = np.mean(ssim_avg)
        std_ssim = np.std(ssim_avg)
        mean_another_ssim_avg = np.mean(another_ssim_avg)
        another_std_ssim = np.std(another_ssim_avg)
        mean_lips_rec_avg = np.mean(lips_rec_avg)
        rec_std_lips = np.std(lips_rec_avg)
        print('testing -  \n mae: %.4f, \n mae_std: %.4f, \n psnr_mean: %.4f, \n psnr_std:%.4f, \n ssim_mean: %.4f, \n ssim_std:%.4f, \n another_ssim:%.4f, \n another_ssim_std:%.4f, \n mean_lips_rec_avg:%.4f, \n rec_std_lips:%.4f' % (mae, mae_std, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_another_ssim_avg, another_std_ssim, mean_lips_rec_avg, rec_std_lips))
        print('%.4f$\pm$%.4f|%.4f$\pm$%.4f|%.4f$\pm$%.4f|%.4f$\pm$%.4f' % (mean_psnr, std_psnr, mean_ssim, std_ssim, mae, mae_std, mean_lips_rec_avg, rec_std_lips))

        zf_mae = np.mean(zf_mae_avg)
        zf_mae_std = np.std(zf_mae_avg)
        zf_mean_psnr = np.mean(zf_psnr_avg)
        zf_std_psnr = np.std(zf_psnr_avg)
        zf_mean_ssim = np.mean(zf_ssim_avg)
        zf_std_ssim = np.std(zf_ssim_avg)
        mean_another_zf_ssim_avg = np.mean(another_zf_ssim_avg)
        another_std_ssim = np.std(another_zf_ssim_avg)
        mean_lips_zf_avg = np.mean(lips_zf_avg)
        zf_std_lips = np.std(lips_zf_avg)
        print('testing -  \n zf_mae: %.4f, \n zf_mae_std: %.4f, \n zf_psnr_mean: %.4f, \n zf_psnr_std:%.4f, \n zf_ssim_mean: %.4f, \n zf_ssim_std:%.4f, \n another_zf_ssim_mean:%.4f, \n another_zf_ssim_std:%.4f, \n mean_lips_zf_avg:%.4f, \n zf_std_lips:%.4f' % (zf_mae, zf_mae_std, zf_mean_psnr, zf_std_psnr, zf_mean_ssim, zf_std_ssim, mean_another_zf_ssim_avg, another_std_ssim, mean_lips_zf_avg, zf_std_lips))
        print('%.4f$\pm$%.4f|%.4f$\pm$%.4f|%.4f$\pm$%.4f|%.4f$\pm$%.4f' % (zf_mean_psnr, zf_std_psnr, zf_mean_ssim, zf_std_ssim, zf_mae, zf_mae_std, mean_lips_zf_avg, zf_std_lips))

        print("Time to reconstruct a slice:", total_time/self.opts.how_many)
