# 33rd place solution - LB 0.758

Competition: asl-fingerspelling
Rank: #33
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434409

## TL;DR
We used a conformer-like model consisting transformer encoder, 1d convolution with CBAM and bi-LSTM.
Overall model size is about 16MB after INT8 quantization.
Training objective is CTC with InterCTC loss.

## Data preprocess
- Used landmarks
    - 1 for nose, 21 for dominant hand, 40 for lips
    - x, y coordinates
- Normalization
    - Standardized distance from nose coordinates
- Feature engineering
    - Concatenation of normalized location, difference of next frame, joint distance of hand
    - Total 582 dims
- Removing data that input frame is shorter than 2 times of target phrase

## Data augmentation
- Horizontal flip landmarks
- Interpolation
- Affine transfrom

## Model
- 2 stacked encoder with Transformer, 1D convolution with CBAM and Bi-LSTM
    - hidden dim: 352
- CTC loss and Inter CTC loss after first encoder
- 17M parameters and INT8 quantization

## Train
- 2 staged training (300 + 200 epochs)
    - Use supplemental and train data in first 300 epochs
    - Use train data 200 epochs
- Ranger optimizer
- Cosine decay scheduler with 12 epochs warmup
- AWP
    - It prevents the validation loss from diverging, but it doesn't seem to improve the edit distance.
    - Used to train long epochs.

## Not worked
- Augmentation
    - Time, spatial, landmark masking
    - Time reverse
- Autoregressive decoder
    - Use joint loss, CTC loss for encoder and Crossentropy for decoder
    - Inference time is longer and the number of parameters larger than those of the CTC encoder alone, but the performance improvement is not clear, so it is not used.
