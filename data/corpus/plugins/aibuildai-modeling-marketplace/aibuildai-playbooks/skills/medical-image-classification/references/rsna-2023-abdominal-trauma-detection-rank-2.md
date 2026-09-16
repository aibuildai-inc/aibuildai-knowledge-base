# 2nd Place Solution

Competition: rsna-2023-abdominal-trauma-detection
Rank: #2
Source: https://www.kaggle.com/c/rsna-2023-abdominal-trauma-detection/discussion/447453

My solution combines knowledge acquired in participating in the previous RSNA challenges, and involves much more than the month I spent working intensively in the competition. I've always enjoyed joining RSNA challenges, and have a special affection for medical imaging because of my relatives' medical profession. 

Although 2nd is a great finish, the conditions in which it happened (i.e. unjustified deadline extension) make it really painful. The Kaggle team still does not understand how much modifying rules last minute hurts participants, or they simply don't care. I was already burnt out by competing full time for a month, adding 2 days on top plus missing first place by nothing is too much for me.

**Updates:** 
- More details added, fixed num_classes mistake.
- Inference code : https://www.kaggle.com/code/theoviel/rsna-abdominal-inf
- **Training code on Github :** https://github.com/TheoViel/kaggle_rsna_abdominal_trauma

## Data 
I use [my datasets] (https://www.kaggle.com/theoviel/datasets?sort=votes)! Give them a quick upvote so I can reach 4x GM. 
In addition, I resize the longest edge to 512 & center crop to 384. I also use 1 frame out of 2 to speed up 2D models inference, and limit stack size to 600. For models requiring a specific input size, images were simply resized afterwards. 
Images are loaded with `dicomsdl` and processed on GPU.   It's fast. The pipeline without ensembling runs in less than 4h. 

## Models

### Overview

Pipeline is below. It has two components: 
-	2D models + RNN, where the frame-level labels are inferred using organ visibility classification when needed. 
-	Crop models for kidney / liver / spleen. Results are fed to the RNN after pooling.

It re-uses winning ideas from the RSNA fracture competition (main references: [[1]](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232), [[2]](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362640), [[3]](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/363232)).

<a href="https://ibb.co/MBBh8wP">[RSNA-Abd-drawio]</a>

### 2D models 

The key to achieve good performance with 2D models is cleverly sampling frames to feed meaningful information and reduce label noise.
To do so, I use a simple but fast `efficientnetv2_rw_t` to infer which organs are present on every frame. During training, frames are sampled the following way:
- kidney / liver / spleen / negative bowel : Pick a random frame inside the organ.
- positive bowel / positive extravasation : Use the frame-level labels.
- Negative extravasation : Sample anywhere

This model extracts probabilities for every 1/2 frame in the stack, and a RNN is trained on top to aggregate results. 

**Details :**
- Heavy augmentations (HFlip, ShiftScaleRotate, Color augs, Blur augs, ElasticTransform) + cutmix (`p=0.5`)
- `maxvit_tiny_tf_512` was best. `convnextv2_tiny` and `maxvit_tiny_tf_384` were also great. 
- Ranger optimizer, `bs=32`, 40 epochs, `lr=4e-5`
- Only 3D info is the 3 adjacent frames used as channels.
- 11 classes : `[bowel/extravasation]_injury`(BCE optimized). And `[kidney/liver/spleen]_[healthy/low/high]`  optimized with the cross entropy.

### Crop models

Strategy is similar : key is to feed to the model crops where the information is located. In that case, I used a 3D `resnet18` to crop the organs, and feed the crop to a 2D CNN + RNN model. It improves performances on kidney, liver and spleen by a good margin. 

**Details :**
- Same augmentations with more cutmix (`p=1.`)
- Ranger optimizer, `bs=8`, 20 epochs, `lr=2e-5`
- Best model uses 11 frames sampled uniformly in the organ. I used different number of frames for ensembling.
- `coatnet_1_rw_224` + RNN was best. I used different heads (RNN + attention, transformers) and other models CoatNet variants for ensembling.
- 3 class cross-entropy loss.

### RNN model

It is trained separately. Its role is to aggregate information from previous models, and optimize the competition metric directly.

**Details :**
- Restrict stack size to 600 (for faster loading), use 1/2 frame (for faster 2D inference). Sequences are then resized to 200 for batching. 
- Heavily tweaked LSTM architecture :
  - 1x Dense + Bidi-LSTM for the 2D models probabilities. Input is the concatenation of the segmentation proba (`size=5`), the classification probas (`size=11 x n_models`), and the classification probas multiplied by the associated segmentation (`size=11 x n_models`)
  - Pool using probabilities predicted by the segmentation model to get organ-conditioned features.
  - Use the mean and max pooling of the `22 x n_models` 2D classification features
  - Independent per organ logits, which have access to the corresponding pooled features. For instance the kidney logits sees only the crop features for the kidney (`3x n_crop_models` fts) , the RNN features pooled using the kidney segmentation, and the `3 x n_models` pooled 2D features for the kidney class.
- AdamW optimizer, `bs=64`, 10 epochs, `lr=4e-5`

### Things that did not work

- YoloX + Ian Pan extravasation boxes. Tried using the data to get crops, and adding the confidence to the RNN model. It did not really help and was painful to implement.
- Adding a sequential head to the first stage worked early on, but as my crop models got stronger I figured out 2D was enough. This allowed for a significant speed up of my inference pipeline which is nice.
- Ensembling did not really help on private ultimately and my best sub is my strongest single model.
- Stuff I tried during the 2 days extended deadline. I was happy with how I managed my time and knew the extension only meant more time for other teams to catch up. Thanks again Kaggle 😭

## Scores : 
- 2D Classification + RNN :
 - Using ConvNext-v2: **Public 0.41** - **Private 0.39**
- Add the crop model:
 - MaxVit (instead of ConvNext) +  CoatNet-RNN : **Public 0.37** - **Private 0.35** (best private)
- Ensemble:
 - 3x2D models, 8x 2.5D models : **Public 0.35** - **Private 0.35**


*Thanks for reading !*
