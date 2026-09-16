# 11th Place Solution

Competition: physionet-ecg-image-digitization
Rank: #11
Source: https://www.kaggle.com/c/physionet-ecg-image-digitization/writeups/11th-place-solution

Overview 
    This solution aims to improve the ​signal-to-noise ratio (SNR)​​ of electrocardiogram (ECG) signals through a streamlined pipeline that combines ​lead reduction, ​advanced network architectures, ​SNR-aware loss functions, and ​post-processing grid alignment.
In the first stage, we adopted hengck23’s excellent notebook, but performed positional optimization on the generated grid points through iterative refinement based on the principle of equal spacing in both x and y directions. Our primary efforts were then devoted to the second stage.
1. Lead Wire Cropping Strategy
      - We use ​4 long lead wires​ cropped from the standard 12-lead ECG.
      - This reduces computational overhead and ​speeds up validation cycles.
      - ​Lead II full​ serves as the reference for evaluating strategy effectiveness.
      -  Cropped images retain clinically relevant information while allowing the use of ​larger neural networks.
2. SNR-Aware Loss Function
      -   An ​SNR-based loss term​ is incorporated into the training objective.
      -   This loss directly optimizes for signal quality, contributing to an ​SNR improvement of ~1.6 dB.
3. Backbone Architecture Upgrade
     -    We replace ResNet34 with ​ConvNeXt Large​ for feature extraction.
     -    This change yields an ​SNR gain of 0.6 dB​ due to better representation capacity.
4.  Iterative Grid Point Alignment
     -    A post-processing step refines detected grid points using an ​iterative spacing correction algorithm.
     -    The method is especially effective for ECG types ​3, 4, 11, and 12, where local distortion is small.
     -    Type 6 also sees minor improvement.
     -    Average SNR improvement: 0.35 dB​ for the supported types.
5. Multi-Scale and Multi-Model Ensemble
    -     Three input sizes are used during training and inference:(2200, 1700), (6600, 1700), (4400, 3400)
    -     Multiple architectures are employed: ConvNeXt Large, EfficientNet, U-Net
6. Effective Strategies
    -    Larger backbone (e.g., ConvNeXt Large)
    -    SNR-aware loss function 
    -    Iterative grid alignment (for types 3, 4, 6, 11, 12) 
    -    Multi-scale training and model ensemble
7.  Ineffective Strategies
    -   Extensive data generation, CycleGAN-based style transfer
    -   We used the relationships I + III = II and AVF + AVR + AVL = 0 to correct the output lead signal values, but unfortunately, the SNR actually decreased.
