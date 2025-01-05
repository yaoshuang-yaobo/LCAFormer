# <p align=center>` LCAFormer: Linear Contour Attention Transformer Network for Lightweight Landslide Detection`</p>


### 1. Overview

In remote sensing imagery, landslides often occur in mountainous terrain where the textures and colors of the landslide and the surrounding landscape are similar. This makes it difficult to accurately delineate the boundaries of the landslide and effectively suppress background interference. In this paper, a Linear Contour Attention Transformer (LCAFormer) model is proposed. By leveraging the complementary characteristics of Convolutional neural networks (CNNs) and Cross-transformer, the model effectively captures both global and local feature information. In the contour extraction stage, a linear contour attention mechanism (LCAM) is used to model the variability of contours across features at different scales. In addition, a multi-feature fusion sampling (MFFS) module is introduced, which restores the resolution of feature maps globally and facilitates the full integration of multi-scale features through a self-attention sampling fusion strategy, further enhancing detection accuracy. LCAFormer, with 1.51M parameters and an inference speed of 2.17G, outperforms 25 lightweight landslide detection models on three landslide datasets, achieving mIoU values of 82.70%, 68.84%, and 90.82%, respectively. Its performance meets the requirements for hardware deployment and can be applied to landslide detection tasks in future work.
