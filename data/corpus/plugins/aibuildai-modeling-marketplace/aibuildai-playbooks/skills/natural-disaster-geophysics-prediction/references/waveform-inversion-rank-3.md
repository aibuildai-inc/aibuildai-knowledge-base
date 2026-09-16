# 3rd place solution

Competition: waveform-inversion
Rank: #3
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587419

Thank you kaggle and the host team for hosting this competition. Congratulations to the winners!


My solution leverages synthetic data generation and Vision Transformer for modeling. Requiring lots of storage and compute resources.
I used more than 13TB of storage and ~5.5 days of `A100 SXM 80G x4`.


## Forward Modeling

I started with [this implementation](https://www.kaggle.com/code/manatoyo/improved-vel-to-seis) by @manatoyo (developed with the help of Google's Gemini). And made the following changes:

1. Change dtype from float64 to float32. This reduced errors a bit. Matlab uses float64 by default, and Gemini knows that. However, OpenFWI seems to be created using a float32 implementation.
2. Generate 6 channels `isx=[120, 137, 154, 155, 172, 189]` to enable fast hflip augmentation.
3. Port to PyTorch and enable batch execution on GPU.

For some vel types, there are large errors in the last row of seismic data between OpenFWI data and generated data, so I zeroed the last row of all data during training.

OpenFWI is not fully open. Their code may have some minor bugs, and if there are serious bugs in their forward modeling code, there may be little we can do about it. Fortunately, the issues don't seem significant at the moment.

## Generating data

I generated the following data for training:
1. 6ch version of Full OpenFWI data.
2. random blending (alpha in [0.3, 0.7]) of two vels from same fold, same vel-type. repeated 8 times with different seeds.
3. random blending (alpha in [0.3, 0.7]) of two vels from same fold, not restricted to same vel-type. repeated 8 times with different seeds.

In my short training runs, generating more data consistently leads to performance improvements.

## Modeling

I started with ViT with RoPE (`eva02_base_patch16_clip_224`) for this task.

1. I first re-aranged the input from (N, 5, 1000, 70) to (N, 5, 250, 280), then interpolated it to (N, 5, patch_size * 18, patch_size * 18).
2. The input is then fed to the ViT model, predicted 16-dim outputs, then pixel-shuffled to (N, 1, 72, 72) and cropped to (N, 1, 70, 70).
3. The logits are transformed to target by `logits.sigmoid() * 6000`.

Then I split ViT layers:
1. First 1/3 layers for intra-channel feature interaction.
2. Average pool 5 channels to 1 channel.
3. Last 2/3 layers for further feature interaction.


## Optimizer

I started with a short `CurveFault_A` only training run. Switching from AdamW to `MuonWithAuxAdam` was a big improvement (9.x -> 7.x). I adapted the implementation from `KellerJordan/Muon` and followed the practices from `MoonshotAI/Moonlight`. Muon was used for weights of all `Linear` layers except patch embedding, for other parameters AdamW was used.

I haven't achieved significant performance improvements with optimizers other than SGD and Adam for many years, so this was a surprise for me, perhaps the biggest takeaway from this competition. The only drawback is that Muon's overhead becomes non-negligible when using small batch sizes.


## Submission

The final model was `eva02_large_patch14_clip_224` trained in 3 stages.

### Stage 1

* HFlip augmentation.
* Input resized to (14 * 18, 14 * 18), 4x output upsampling.
* 10 epochs on original data and generated data
* 50% constant learning rate + 50% cosine decay learning rate

MAE on 20% validation data: 8.1

### Stage 2

* Resume from stage 1
* HFlip augmentation.
* Input resized to (14 * 35, 14 * 35), 2x output upsampling.
* 1 epochs on original data and generated data
* 50% constant learning rate + 50% cosine decay learning rate

MAE on 20% validation data: 7.6

### Stage 3

* Resume from stage 2
* HFlip augmentation.
* Input resized to (14 * 35, 14 * 35), 2x output upsampling.
* 1 epochs on validation data
* cosine decay learning rate

More synthetic data and more epochs could further improve the performance, this was the best I could finish with my GPU hours.

Submission was a 20%+30%+50% blending of models from 3 stages. With `x.clip(1500, 4500).round()`.

**Update**
I made a mistake in the inference script by not zeroing the last row, which is inconsistent with training. After fixing the bug, the score is (Public 7.96, Private 7.99).
Fortunately, it doesn't affect the ranking.

## Code

[Submitted file](https://www.kaggle.com/datasets/tascj0/fwi-submission)
[Inference notebook](https://www.kaggle.com/code/tascj0/fwi-inference)
[Training code](https://github.com/tascj/kaggle-waveform-inversion)
