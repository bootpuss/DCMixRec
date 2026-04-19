import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import DCMixRec
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])
    network = nn.DataParallel(network, device_ids=gpu_ids)

    return network


def get_generator(name, opts):

    if name == 'DCMixRec':
        ic = int(opts.use_prior) * 2
        if opts.use_prior == 1:
            network = DCMixRec(n_channels=ic, G0=48, kSize=3, D=3, C=4, G=48, dilateSet=[1,2,3,3])
        elif opts.use_prior == 2:
            network = DCMixRec(n_channels=ic, G0=64, kSize=3, D=3, C=4, G=64, dilateSet=[1,2,3,3])

    else:
        raise NotImplementedError

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters: {}'.format(num_param))
    return set_gpu(network, opts.gpu_ids)
