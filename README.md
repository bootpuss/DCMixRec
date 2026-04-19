# DCMixRec
Detail-Contour Feature-Disentangled Mixed Heterogeneous Network for Multi-Contrast MR Imaging Reconstruction


## Abstract
Multi-contrast MRI provides superior tissue contrast for diagnosis, yet its clinical utility is hampered by long acquisition times. Deep learning enables accelerated reconstruction, but existing methods suffer from two limitations: (i) architectural homogeneity: they adopt either CNNs for local features or Transformers/Mamba-based models for long-range dependencies, without leveraging the complementary benefits of heterogeneous architectures; (ii) frequency-blinded cross-modality fusion: existing methods indiscriminately mix high-frequency details with low-frequency global anatomical structures, causing interference that hinders clear edge and texture recovery, thereby limiting reconstruction quality. To address both challenges, we propose DCMixRec, a **D**etail-**C**ontour Feature-Disentangled **Mix**ed Heterogeneous Model for accelerated multi-contrast MR Imaging **Rec**onstruction. Specifically, the Detail-Contour Feature-Disentanglement (DCFD) encoder factorizes features into complementary contour and detail subspaces to be processed in parallel, subsequently integrating them into a unified representation. The State-Gated Dense Reconstruction (SGDR) decoder then processes the unified features via a dynamically constrained architecture, leveraging topological states to gate the spatial decoding process. Extensive experiments on the fastMRI knee and SIMON brain datasets demonstrate that DCMixRec consistently outperforms state-of-the-art methods in reconstruction quality and anatomical fidelity. Furthermore, t-SNE and feature activation maps explicitly validate the detail-contour feature disentanglement in latent space, confirming the interpretability of our proposed disentanglement mechanism.

## Environment Setup

We recommend using [Anaconda](https://www.anaconda.com/) or Miniconda to manage the Python environment.

```bash
conda create -n dcmixrec python=3.10 -y
conda activate dcmixrec
pip install -r requirements.txt
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --experiment_name 'SIMON_t2_G1D30' --dataset 'SIMON' --net_G 'DCMixRec' --n_recurrent 5 --how_many_valid 96 --dataroot /path/to/your/dataset.csv --loadSize 256 --fineSize 256 --output_nc 2 --input_nc 2 --batch_size 1 --mask G1D30 --use_prior 2 --gpu_ids 0 --n_epochs 100 --num_workers 8
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --experiment_name 'SIMON_t2_G1D30' --dataset 'SIMON' --net_G 'DCMixRec' --n_recurrent 5 --results_dir EXP_results/ --how_many 144 --dataroot /path/to/your/dataset.csv --loadSize 256 --fineSize 256 --output_nc 2 --input_nc 2 --mask G1D30 --use_prior 2 --resume './EXP_Outputs/SIMON_t2_G1D30/checkpoints/latest.pt' --num_workers 8
```
