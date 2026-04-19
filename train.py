import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser(description='DCMixRec')

# model name
parser.add_argument('--experiment_name', type=str, default='test', help='give a experiment name before training')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

parser.add_argument('--dataset', type=str, default='Cartesian', help='dataset name')   # SIMON / fastmri

# model architectures
parser.add_argument('--net_G', type=str, default='DRDN', help='generator network')
parser.add_argument('--n_recurrent', type=int, default=5, help='Number of reccurent block in model')
parser.add_argument('--use_prior', type=int, default=1, help='use prior (integer value)')

# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')

# training options
parser.add_argument('--n_epochs', type=int, default=100, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=1, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=1, help='save evaluation for every number of epochs')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# loss
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--log_freq', type=int, default=10, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./EXP_Outputs', type=str, help='Output path.')

parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')

# mask
parser.add_argument('--mask', type=str, default="G1D30", help='mask of trajectory')

parser.add_argument('--how_many_valid', type=int, default=50, help='how many valid images to run')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--loadSize', type=int, default=266, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')

opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")


def sigtoimage(sig):
    x_real = torch.unsqueeze(sig[:, 0, :, :], 1)
    x_imag = torch.unsqueeze(sig[:, 1, :, :], 1)
    x_image = torch.sqrt(x_real * x_real + x_imag * x_imag)
    return x_image


cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

train_set, val_set, test_set = get_datasets(opts)

train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
# output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
output_directory = os.path.join(opts.output_path, opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)

# training loop
best_psnr, best_ssim = 0.0, 0.0
for epoch in range(ep0, opts.n_epochs + 1):

    train_bar = train_loader
    
    model.train()

    model.set_epoch(epoch)

    total_steps = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{opts.n_epochs}] - PSNR: 0.000, SSIM: 0.000", dynamic_ncols=True)
    for it, data in enumerate(train_bar):

        # if total_steps >= 10:
        #     break

        total_steps += opts.batch_size
           
        total_iter += 1
        model.set_input(data)
        model.optimize()

        if it % opts.log_freq == 0:
            with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(model.get_current_losses().values())
        
        # save checkpoint
        if (total_steps) % opts.save_latest_freq == 0:
            checkpoint_name = os.path.join(checkpoint_directory, 'latest.pt')
            model.save(checkpoint_name, epoch, total_iter)


        metric_img_rec = model.recon
        metric_img_real = model.t2_image_full

        metric_img_rec_one = sigtoimage(metric_img_rec)
        metric_img_real_one = sigtoimage(metric_img_real)

        fake_im = metric_img_rec_one.cpu().data.numpy()
        real_im = metric_img_real_one.cpu().data.numpy()

        psnr_list = []
        ssim_list = []
        for j in range(real_im.shape[0]):
            real_im_ = real_im[j, ...]
            fake_im_ = fake_im[j, ...]

            psnr_list.append(psnr(real_im_, fake_im_, data_range=1))
            ssim_list.append(ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, channel_axis=2))

        train_bar.set_description(f"Epoch [{epoch + 1}/{opts.n_epochs}] - PSNR: {np.mean(psnr_list):.3f}, SSIM: {np.mean(ssim_list):.3f}")


    model.update_learning_rate()

    if (epoch+1) % 10 == 0:
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)


    # evaluation
    print('Validation Evaluation ......')

    if (epoch+1) % opts.eval_epochs == 0:

        combined_img_path = os.path.join(image_directory, 'combined_{:03d}.png'.format(epoch))
    
        if opts.wr_L1 > 0:

            vis_pred = (model.recon.detach()[:, 0:1, :, :] ** 2 + model.recon.detach()[:, 1:2, :, :] ** 2).sqrt()
            vis_gt = (model.t2_image_full.detach()[:, 0:1, :, :] ** 2 + model.t2_image_full.detach()[:, 1:2, :, :] ** 2).sqrt()
            vis_input = (model.t2_image_sub.detach()[:, 0:1, :, :] ** 2 + model.t2_image_sub.detach()[:, 1:2, :, :] ** 2).sqrt()
        
            combined_vis = torch.cat([vis_input, vis_gt, vis_pred], dim=3)
        
            save_image(combined_vis, combined_img_path, normalize=True, scale_each=True, padding=5)

        current_psnr, current_ssim = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            current_psnr, current_ssim = model.evaluate(val_loader,epoch)

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            checkpoint_name = os.path.join(checkpoint_directory, 'model_psnr.pt')
            model.save(checkpoint_name, epoch, total_iter)

        if current_ssim > best_ssim:
            best_ssim = current_ssim
            checkpoint_name = os.path.join(checkpoint_directory, 'model_ssim.pt')
            model.save(checkpoint_name, epoch, total_iter)

