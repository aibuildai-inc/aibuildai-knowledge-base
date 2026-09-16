# 17th Solution: Conformer + CTCLoss +  500 epoch training

Competition: asl-fingerspelling
Rank: #17
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434364

# TL; DR

Fixed-length input (220 frames), padding shorter inputs and resizing longer inputs.

2 layer MLP landmark encoder + 6 layer 384-dim Conformer + 1 layer GRU and 500 epochs training (takes ~8 hours on Kaggle TPUs).

Post-processing (+0.003 in CV and LB) by:
```
if len(pred) <= 4:
    pred = pred + " -aero"
```

My notebook is available at https://www.kaggle.com/code/nightsh4de/ctc-transformer/notebook
It is primarily based on the previous 1st solution https://www.kaggle.com/code/hoyso48/1st-place-solution-training and some public notebooks of this competition:
 https://www.kaggle.com/code/markwijkhuizen/aslfr-transformer-training-inference#Landmark-Embedding
https://www.kaggle.com/code/irohith/aslfr-transformer 
https://www.kaggle.com/code/shlomoron/aslfr-ctc-on-tpu. Many many thanks to them.

# Data Preprocessing and Augmentation

Basically the same as the previous 1st solution. But I found using 3d positions (means including depth) and pose landmarks gives better CV and LB score.

My input: left-right hand, eye, nose, lips and pose landmarks.

# Model

I believe this task would be quite similar to Automatic Speech Recognition (ASR), so I used Conformer https://arxiv.org/abs/2005.08100

# Post-preprocessing

1. I checked my worst predictions in the validation set and found that shorter predictions are worse. And most very short predictions (length less than 5) are basically predicting nothing, e.g. single characters like "a" or space " ".
2. I found that only few label's length is less or equal to 5. 
3. So adding some make-up phrases to very short predictions might be a good idea.
3. I picked the most common chars in training set: "a", "e", "r", "o", "-", " ".
4. I test all the combinations of "a", "e", "r", "o", "-", " " based on validation set and " -aero" is the best.

# What not worked

1. Autoregressive transformer models. 

I spent most of my time on transformers. But they strongly overfits the phrase and I couldn't find a way to solve it.

The problem is , for an input X[1...n] and its phrase "123456789". We may expect our model predict "12345" for the first half input X[1...n/2]. 

For CTC models, it is true. But my autoregressive model always makes half correct prediction "12345" + half random incorrect prediction e.g. some random 5-digit numbers.

2. Masking for variable-length input

I tried to use the same masking techniques as the previous 1st solution https://www.kaggle.com/code/hoyso48/1st-place-solution-training, to support longer input frames.

However, masking models always produce lower CV and LB scores (-0.02). I believe it is due to the masking limits the output length, but shorter inputs still require a longer output space. Although, unfortunately, I don't have time to verify it, I believe it might enable much better solutions.

3. Empty embedding for fixed-length input

I tried to use learnable constant weights for padded empty frames and 2-layer MLP for encoding landmarks in original frames. It seems to me makes more sense, but CV score is worse than direct encoding all frames.
