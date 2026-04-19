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
parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
parser.add_argument('--loadSize', type=int, default=266, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
parser.add_argument('--results_dir', type=str, default='./EXP_results', help='saves results here.')


opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {} \n'.format(num_param))

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

output_directory = os.path.join(opts.output_path, opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

print('Test Evaluation ......')
model.eval()
with torch.no_grad():
    model.test(test_loader)