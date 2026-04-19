from datasets.aligned_dataset_simon import AlignedDatasetSIMON
from datasets.aligned_dataset_fastmri_knee import AlignedDataset2 as AlignedDataset2_fastmriknee

def get_datasets(opts):

    if opts.dataset == 'SIMON':
        opts.phase = 'train' 
        trainset = AlignedDatasetSIMON(opts)
        opts.phase = 'valid' 
        valset = AlignedDatasetSIMON(opts)
        opts.phase = 'test' 
        testset = AlignedDatasetSIMON(opts)

    elif opts.dataset == 'fast-mri-knee':
        opts.phase = 'train' 
        trainset = AlignedDataset2_fastmriknee(opts)
        opts.phase = 'valid' 
        valset = AlignedDataset2_fastmriknee(opts)
        opts.phase = 'test' 
        testset = AlignedDataset2_fastmriknee(opts)

    else:
        raise NotImplementedError

    return trainset, valset, testset