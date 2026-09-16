# 21st place solution - FP co-teaching with loss improvements

Competition: rfcx-species-audio-detection
Rank: #21
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220436

Congratulations to the top finishers!

The whole my solution is here: https://github.com/MPGek/mpgek-rfcx-species-audio-detection-public.
I left the best configs that were used in the final submission.

## Summary
Summary of my solution:
- General augmentations with frequency band augmentations:
 - Frequency band filtering
 - Frequency band mixup
- TP training with a combined loss of the BCE for confident samples and LSoft 0.7 for noisy samples
- FP training with a BCE for confident samples and ignored losses for samples with noisy samples
- FP co-teaching training with loss as described in the co-teaching paper, but with the extra loss for the high loss samples
- Ensemble of TP, FP, FP co-teaching results.

## Spectrogram and random crop
For the training, I used a random crop with the centered sample in 10 seconds and a random crop for 6 seconds.
For validation and prediction, I used 6 seconds crop with a stride in 2 seconds with maximizing outputs.
For the mel spectrogram, I used the following parameters:
- mels count: 380
- FTT: 4096
- window length: 1536
- hop length: 400
- fmin: 50
- fmax: 15000

So one sample in 6 seconds produced an image with size 720 x 380 pixels

## Augmentations
**Augmentation that improves LB:**
- Gaussian noise
- Random crop + resize with size reduction in 40 and 20 pixels
- Frequency band filtering - based on f_min and f_max of the sample I set 0 values to all mels that lower or higher than f_min and f_max with a sigmoid transition to remove sharp edges
 - Frequency band mixup - for some samples I used frequency band filtering and then I mixed it with different samples with a band that higher than f_max and with a band that lower than f_min. So I managed to get a single image with the 3 mixed samples.

**Example of the Frequency band filtering (top - original sample, bottom - sample after filtering):**

**Example of the Frequency band mixup (top - original sample, bottom - sample after mixup):**

**Augmentation that doesn't improve LB:**
- SpecAugment
- Sum mixup

## Network topology

I got the best results on EfficientNetB2, B4, B7 (noisy students weights) with a simple FC to 24 classes after the adaptive average pool.

I tried to use different head but all of them gave the same or worse result:
- the hyper column
- the hyper column with auxiliary losses after each pooling
- extra spatial and channel attentions blocks - CBAM
- dense convolutions

## TP training
Based on the post where described that every sound file can present unlabeled samples I have to work noisy samples.

I split all samples into confident samples and noisy samples:
- confident samples - all sigmoid outputs for classes which present in the train_tp.csv file (1 in targets tensor)
- noisy samples - all sigmoid outputs for classes which not described in the train_tp.csv file (0 in target tensor)

For the confident samples, I used simple BCE. For the noisy samples, I used LSoft with beta 0.7 (https://arxiv.org/pdf/1901.01189.pdf). 
LSoft:
```
def forward(self, input: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            pred = torch.sigmoid(input)
            target_update = self.beta * target + (1 - self.beta) * pred
        loss = F.binary_cross_entropy_with_logits(input, target_update, reduction=self.reduction)
        return loss
```

In the loss function, I flatten all outputs (even batch dim) to a linear array. And split items into 2 arrays where targets were 1 and where targets were 0.

With LSoft I got on EfficientNetB7 0.912-0.915 LB.
Without LSoft - BCE for all samples I got only about 0.895-0.900.

## FP training
For the FP training, I used a dataset with undersampling of the FP samples. Each epoch had all TP samples and the same count of the FP samples.
I used batch sampling to provide a balanced batch of TP/FP samples - after each TP I added FP with the same species id.

In the loss function, I calculate loss only for those sigmoid outputs that present in train_tp.csv or train_fp.csv. So all noisy samples are ignored.

## FP co-teaching training
I have tried to find a way how to use FOCI or SELFIE to work with noisy data, but all of them use historical predictions of each sample. With my random crop and frequency band mixup it's almost impossible. Even shift for 0.5-1 seconds can add a new species to the sample. So historical data will be incorrect.

I tried co-teaching training because it doesn't require historical data.
Paper: https://arxiv.org/pdf/1804.06872.pdf
Code sample: https://github.com/bhanML/Co-teaching

When I implemented co-teaching training I have only 5 days before the deadline.
The first experiments with co-teaching gave 0.830 LB for the TP and 0.880 LB for the FP. So it looked like a bad experiment.

I tried to improve loss function by adding high loss samples with changed targets (by default co-teaching should ignore high loss samples as wrong samples).

The final loss function consists of:
- 50% lowest loss samples (all confident samples mandatory added to this part of loss with scale factor 2)
- 45% ignored losses
- 5% highest loss samples with the changed target (1 for predictions with sigmoid >= 0.5 and 0 for sigmoid < 0.5)
The loss implementation is here: https://github.com/MPGek/mpgek-rfcx-species-audio-detection-public/blob/main/model/forward_passes_coteaching.py

When I invented this loss I had only 3 days before the deadline.
This loss has a good potential for future experiments. I trained only 2 experiments with this loss, both with the same hyperparameters. The first one had a bug so it produces extremely high metrics and used wrong epochs to submission - however, even with the bug it produces a good LB score comparing to the original FP training.

## Folds and best epoch metric
To chose the best epoch in all training I used the BCE which calculated only on confident samples.
In some experiments I used 5 stratified KFold, in some, I used 7 folds with stratified shuffle split with test size 0.3.

## Ensembles
The TP training with EfficientNetB7 gave me only 0.912-0.915 on the public LB.
The FP training with EfficientNetB2-B4 gave only 0.887-0.893 LB.
The ensemble of the TP and FP gave 0.929 LB.

The FP co-teaching training on simple EfficientNetB2 gave me 0.925 LB (a quite good improvement from the original FP with 0.893)

**The final ensemble consists of all best experiments (0.941 public LB and 0.944 private LB):**
- TP EfficientNetB7 with 0.915
- FP EfficientNetB2-B4 with 0.893
- FP co-teaching EfficientNetB2 with 0.925
