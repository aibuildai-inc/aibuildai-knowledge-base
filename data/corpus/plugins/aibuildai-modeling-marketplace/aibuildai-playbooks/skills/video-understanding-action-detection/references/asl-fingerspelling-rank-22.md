# 22nd place solution: CTC Loss, Strong augmentations, CNN+MHSA

Competition: asl-fingerspelling
Rank: #22
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434680

[My best-performing model](https://github.com/SamratThapa120/sign-language-finger-spelling/blob/master/signet/models/feature_extractor_downsampled.py) ([config](https://github.com/SamratThapa120/sign-language-finger-spelling/blob/master/signet/configs/ctc_loss_with_downsampled_deploy_concataug.py))is based on the [1st place solution of the previous competition](https://www.kaggle.com/code/hoyso48/1st-place-solution-training/notebook) by @hoyso48. I trained the model using vanilla CTC Loss, and used greedy decoding for inference. 

### Things that worked for me:
- **Longer input frames length:**  A longer input frame length of 384 performed better than shorter length of 256.   
- **Deeper/larger model:**  My best-performing model has 7 blocks with hidden dimension of 256. It has 9.5M parameters.
- **Pose keypoints:** Adding pose information was helpful. Best-performing model uses pose+hands+lips+eyes.
- **CNN+MHSA:** CNN+MHSA model > Only CNN Model>Only MHSA model
- **Strong augmentation:** My model was performing well based on local CV, and public LB, with strong correlations between the two.But when I tested the model using this [Gradio app](https://github.com/SamratThapa120/gradio-ASL-fingerspelling-recognition) using my webcam, it could not recognize my signs, so I had to use strong data augmentations to get the model to work. Especially temporal mask helped because I sign slower than the pros. These augmentations also boosted the public LB score by +0.006.

    flip_lr_probability=0.5
    random_affine_probability=0.75
    freeze_probability=0.5
    temporal_mask_probability=0.75
    temporal_mask_range=(0.2,0.4)

- **concat augmentation:** Randomly concatenate two short landmark sequences, as well as their labels. This improved public LB by +0.008. I applied this augmentation to 40% of all training samples.

### Here are other things I tried, that did not contribute to the best-performing model:

- **transformer-encoder+decoder:** Transformer endoder-decoder model with cross-entropy loss.

- **transformer-decoder:** CNN+MHSA model and Transformer-like decoder with cross-entropy loss.

- **Causal-masking in self-attention:** removing causal masking performed better than using causal mask.

- **Attention-span in self-attention:** I thought that there would not be long-term dependency between frames for this task, so I tried to reduce the attention span of self-attention. Although there was no performance degradation, it did not boost performance either.

- **Focal loss**: I tried the CTC Focal loss based on [this](https://github.com/TeaPoly/CTC-OptimizedLoss/blob/main/ctc_focal_loss.py) repo, but there were no gains.

- **Erase landmarks augmentation**: Randomly erase landmarks except the hand landmarks, to make the model more robust to mediapipe's detection errors.

Originally I was using [pytorch](https://github.com/SamratThapa120/sign-language-finger-spelling/tree/pytorch), but I switched to Tensorflow because I faced several issues when converting the pyTorch model to tfLite. However, training with CTC Loss was upto 10x slower in tensorflow than pytorch. Looking back, I think I should have put more effort into fixing the model conversion issue, as I would have been able to perform more experiments.  

Congratulations to the winning teams. I would also like to thank Google for hosting this competition, it was a valuable opporunity to learn many new things. Also, I would like to thank everyone who shared their notebooks and unique ideas.
