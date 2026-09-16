# 32nd place solution - what worked for me

Competition: rfcx-species-audio-detection
Rank: #32
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220322

Congratulations to the top finishers!

This was my first encounter to audio competition, so I tried a lot of maybe implausible ideas and learned a lot. Especially, solutions from Cornell Birdcall Identification and Freesound Audio Tagging 2019 were helpful.

My finish is not strong, but I wanted to share some of the things that I *believe(not sure since my score is not sufficiently high)* worked for me (increased cv or public score), and hear what other kagglers experienced.

* Frequency Crop

  * For one audio clip, crop 24 different crops according to fmin&fmax of each species.
  * I believe it is similar to what @cpmpml did and @hengck23 did(without repeating convolution computations)

* LSoft

  * I used only TP and FP crops as labels. For example, for each row in TP or FP, only 1 out of 24 labels is present.

  * Based on BCELoss, I used 'lsoft' for unknown labels. LSoft is introduced kindly by @romul0212 at https://github.com/lRomul/argus-freesound/blob/master/src/losses.py

  * This is my loss computation with LSoft. `mask` indicates where the label is known. `true` for unknown labels is initialized with 0.

    ```python
    tmp_true = (1 - lsoft) * true + lsoft * torch.sigmoid(pred)
    true = torch.where(masks == 0, tmp_true, true)
    loss = nn.BCEWithLogitsLoss()(pred, true)
    ```

* Iterative Pseudo Training

  * Since train set is only very sparsely annotated, I thought re-labeling with the model then re-training will help, and it indeed helped. I pseudo trained for 3 stages.
  * When pseudo training, I didn't use LSoft and used vanilla BCE.

* LSEPLoss

  * Our metric is LWLRAP, so it is important to focus on rank between labels for each row. I used LSepLoss which fits this purpose, which is introduced kindly by @ddanevskyi at https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/97926

  * After stage3 of BCE pseudo training, I pseudo trained extra 2 stages with LSEPLoss

  * I fixed original code a bit to allow soft labels.

    ```python
    def lsep_loss(input, target):
        input_differences = input.unsqueeze(1) - input.unsqueeze(2)
        target_differences = target.unsqueeze(2) - target.unsqueeze(1)
        target_differences = torch.maximum(torch.tensor(0).to(input.device), target_differences)
        exps = input_differences.exp() * target_differences
        lsep = torch.log(1 + exps.sum(2).sum(1))
        return lsep.mean()
    ```

    

* Global Average Pooling on only positive values

  * We need to know if the species is present or not. We don't care if it appears frequently or not. I thought doing global average pooling on whole last feature map of CNN will yield high probabilities for frequent occurrences of birdcall in one clip and low probabilities for infrequent occurrences, which doesn't match our goal. So I took mean of only positive values from the last feature map of CNN.

  * Following code is attached at the end of CNN's extracted feature map

    ```python
    mask = (x > 0).float()
    features = (x*mask).sum(dim=(2, 3))/(torch.maximum(mask.sum(dim=(2, 3)), torch.tensor(1e-8).to(mask.device)))
    ```

* Augmentations

  * Gaussian/Pink NoiseSNR, PitchShift, TimeStretch, TimeShift, VolumeControl, Mixup(take union of the labels), SpecAugment
  * Thanks to @hidehisaarai1213 for kindly sharing https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english

* One inference on full clip

  * I didn't resize the spectrogram, so I was able to train on crops and infer on full image.
  * When we don't resize, due to the property of CNN, I believe doing sliding windows prediction on small crops is just an approximation for doing one inference on the full image.

* Validation - only use known labels

  * I did validation clip-wise, only on TP and FP labels. From the prediction, I removed all values corresponding to unknown labels, flattened, then calculated LWLRAP. It correlated with LB quite well on my fold0

My baseline was not so strong(~0.8), so I might had fundamental mistakes in my baseline.
I achieved 0.927 public with efficientnet-b0 fold0 3seed average, but my score worsened when doing 5fold ensembling. I tried average, rank mean, scaling on axis1 then taking mean, calculating mean of pairwise differences taking average, but it didn't help.
I'm planning to study top solutions to find out what I missed

I'd really appreciate it if you share some opinions with my approaches and things that I missed.
