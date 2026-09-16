# 6th Place Solution Summary

Competition: waveform-inversion
Rank: #6
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587460

First, I’d like to thank the competition organizers and the open-source community—your work and shared code were invaluable in helping me learn and improve.


**1. Synthetic Training Data Generation**

To augment the dataset, I generated approximately 10 million additional training samples. To optimize storage efficiency, inputs were downsampled to dimensions (5, 500, 70) and stored in np.float16 format. The synthetic data consisted of:

40% Gaussian noise/random rotations/scaling/shifts

40% Mixup-augmented samples

20% Samples generated via Denoising Diffusion Implicit Models (DDIM)

Code references:
[Improved VEL to SEIS]( https://www.kaggle.com/code/manatoyo/improved-vel-to-seis/notebook)
[Waveform Inversion (VEL to SEIS)](https://www.kaggle.com/code/jaewook704/waveform-inversion-vel-to-seis)

**2. Model Architecture**

ConvNeXtV2-Base (convnextv2_base.fcmae_ft_in22k_in1k_384)

CaFormer-B36 (caformer_b36.sail_in22k_ft_in1k_384)

Modifications were adapted from:

[CaFormer Full-Resolution Improved](https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved)
[ConvNeXt Full-Resolution Baseline]( https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline)

**3. Training Strategy**
*Stage 1 (Primary Training)*

Trained for 60 epochs on the synthetic dataset.

Cross-validation (CV) scores:

CaFormer: 9.2/ConvNeXtV2: 10.2/Ensemble: 8.1

Leaderboard (LB) / Privateboard  (PB): 9.9 / 10.0


*Stage 2 (Test-Time Fine-Tuning)*

Further fine-tuned for 2 epochs on synthetic test-derived data.

Generated additional training samples via:

Forward modeling predictions on test data (y → x).

Applied horizontal flipping (HFlip) for augmentation.

Total synthetic data: 300k samples (from 75,818 test/validation samples × 2 models × 2 flips).

Resulting CV scores:

CaFormer: 8.6/ConvNeXtV2: 9.0/Ensemble: 7.6
Leaderboard (LB) / Privateboard  (PB): 8.8 / 8.9

**4. Late-Stage Insights**

Discovered Stage 2’s significant impact in the final hours (~1.1 LB improvement).

Training was halted 30 minutes before the deadline—further data generation and training could likely have yielded additional gains.
