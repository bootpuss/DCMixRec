# DCMixRec
Detail-Contour Feature-Disentangled Mixed Heterogeneous Model for Multi-Contrast MRI Reconstruction.

## Introduction
Magnetic Resonance Imaging (MRI) plays a critical role in clinical diagnosis by revealing structural and pathological details through tissue contrast. However, multi-contrast MRI protocols often suffer from long acquisition times, which limit their practical utility. While deep learning has shown great promise in accelerating MRI reconstruction, existing approaches are typically constrained to a single architectural paradigm—such as CNNs for local feature extraction, or Transformers and Mamba for capturing long-range dependencies. A unified framework that effectively integrates these complementary models remains largely unexplored.

To bridge this gap, we propose DCMixRec, a novel Detail-Contour Feature-Disentangled Mixed Heterogeneous Model for accelerated multi-contrast MRI Reconstruction. Our framework pioneers the synergistic integration of state-space, self-attention, and convolutional models to achieve computationally efficient and context-aware reconstruction.

Extensive evaluations on the SIMON​ and fastMRI​ datasets demonstrate that DCMixRec achieves superior reconstruction quality—both quantitatively and visually—outperforming existing mono-contrast and multi-contrast methods. It excels in preserving anatomical details and structural consistency, especially under low-sampling scenarios, and shows strong generalizability across various anatomical regions and sampling protocols.
