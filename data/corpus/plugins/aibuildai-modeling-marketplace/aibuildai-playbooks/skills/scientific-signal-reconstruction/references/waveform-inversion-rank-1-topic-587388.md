# 1st Place Solution

Competition: waveform-inversion
Rank: #1
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587388

First and foremost, a sincere thank you to the competition organizers for this challenge, and congratulations to all my fellow competitors on their impressive work.

Crazy how nobody moved a single place up or down in top 28.


Goals List:
✔️Single digit submissions solo gold
✔️Solo win a competition #1

Anyways, here's what I have done:

# **Preprocessing**

This is not a segmentation task. The velocity model pixels don't line up spatially with the seismic data.
An early experiment tested it. I had a baseline that resized the input from the original (5, 1000, 70) to (5, 350, 70) and got a CV of 46.9. , I reshape the input to (1, 350, 350), basically forcing the channels into a spatial layout. CV jumped to 32.5

This is how the non-resized 1x1000x350 input looks like:



# **Architecture**

Unet didn’t make sense to me, my input was 350x350, the pixel values of the velocity model did not spatially align with the input, so it did not make much sense to me to go back to previous layers to get intermediate layer’s embeddings like a Unet does

Anyways, it turns out, vision transformers don’t necessarily need a Unet to do their bidding 🙂

Vision transformers are secretly segmentation-like regression models:



This image size was a nice square so it was easy to work with and scale.
I started using this without a decoder with the best vision transformer I found, EVA02-small model. The performance was really cool, even at this stage it got me a close to 30 score on CV with just 40 epochs of training.

To scale up the model, I experimented with base and large variants which were scoring better, but what was scoring even more was repeating the architecture as an encoder+decoder setup:



Training this scored CV 28, repeating this (encoder+decoder) with base or large variant of the model did not give improvement and were overfitting

Training EVA, it was still 20-30% slower than training the original ViT, I found a better backbone after experimenting to find out what was making EVA score much better than ViT, Depth? Channels? Number of attention heads? MLP layer scaling? Gated activations? Turns out, RoPE (Rotary Positional Embeddings) was contributing to the majority of the bottleneck, then I chose better pretrained weights and I got vit_small_patch14_reg4_dinov2.lvd142m as my ideal choice of backbone. On the same training, ViT+RoPE got me under 26 MAE.

Here is the final architecture:



Code: https://www.kaggle.com/code/harshitsheoran/yale-fwi-vit-architecture?scriptVersionId=248201360

My loss function is MAE loss, I apply sigmoid on my logits and scale them in range of (1500, 4500)


# **Training**

Pretty much every training run used the same recipe:
- LR: 1e-4 with cosine annealing down to ~1e-5
- Optimizer: AdamW
- Loss: MAE
- HFlip and TTA
- EMA

I used 440k out of 470k for training, 30k for validation, the core of my training at MAE of 26 with 50 epochs is that I am severely underfitting, increasing image size and restarting training can lead to much better scores, for a total of 300 epochs:



This model was done training on May 28 (more than a month before the deadline), the last month was done exploring the next part of the solution.

# **Generating Data**

Data Augmentation is a key part of my solution, before an almost perfect replication of the forward modeling function was available, I used forward-modeling + denoiser model setup where denoiser model was the architecture above designed to predict the noise still left in the forward-modeled data

After failing experiments with generating new velocity models with heuristics or diffusion, I went ahead with using FiveCrop and 5xRandomAffine augmentations {RandomAffine(p=1.0, degrees=15, translate=(0.2, 0), shear=(-15, 15), padding_mode='reflection', resample='nearest')} on existing velocity models to generate more velocity models, this gave me 10x more data of 4.7 million seismic samples, I use this data for pretraining on every step, the training graph looks like this:



Later in the competition, a perfect method for forward modeling was publicly available, at the time, the original notebook was running at a speed of about 10 images per minute which was unusable, using pytorch compile and a 5090, I sped it up to >5000 images per minute per gpu which is a lot more bearable for the next time consuming step.

I call it “Iterative Pseudo” where after every epoch, I predict on my validation and competition’s test set, about ~95k samples in total, this prediction is a pseudo prediction (for the test set) as we do not know how good or bad it is, I take the predicted velocity models and forward-model them to generate what should be the input for them and use this input in training the next epoch. This step is repeated for every epoch, this does add extra time to each epoch but the payout in terms of score is so worth it.




As pretraining epochs are 10x larger, all epochs weighted, this was trained for a total of 570 epochs.

Another observation was made that most of the new score that the model is improving for a while, has been coming from mainly 4 out of 10 methods. So, model was further trained with removing the data that doesn’t come from [‘CurveFault_B’, ‘CurveVel_B’, ‘Style_A’, ‘Style_B’], this removes more than half of the data, speeds up training, frees up the model’s parameters to focus more on this part.

Then for inference, I would replace the predictions that were from these 4 methods with the predictions from the model below, which yielded a score of 7.5 on CV, 7.0/7.0 on Public/Private LB



Training this with larger image size at the end, with 896x896 and ensembling that Top4 model to the existing Top4 model yields my best submission at 7.28 CV, 6.9/6.9 on Public/Private LB

The whole process from start to finish takes about 15 days on 4x 5090.

My solution does not feature much for ensembling, and can be cleanly trained for longer to generate a hopeful sub 5 leaderboard.

# **That didn’t work / Didn’t get to try**

MAE/MIM self-supervised pretraining
Standard Augmentations
Unet decoder, UperNet, Mask2Former, ViT-Adapter, Fusion heads, more complicated heads than a linear layer

Due to being short on time, I did not go back and recreate the pretraining data with perfect simulation, that could still improve the score more.

Idea that I didn’t get to try: Prediction from the model forward modeled to generate simulated seismic input - original seismic input to compute gradient, the original seismic input, this gradient and the predicted velocity model as input to a new model that will refine the prediction according to the gradient



Thank you for reading!
