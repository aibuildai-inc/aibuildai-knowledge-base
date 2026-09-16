# 13th place solution

Competition: rsna-2024-lumbar-spine-degenerative-classification
Rank: #13
Source: https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/539510

# 2 stage system

## Summary

Similar to other competitors, we implemented a 2-stage system composed of three core models (plus an additional one for pretraining):

- Keypoints: Localizing intervertebral spots.
- Levels: Classifying axial images into spinal levels.
- Pretraining (Patch): Classifying individual patches.
- Sequence: A 2D model combined with a Transformer for final predictions.

For detailed training specifics, please refer to the accompanying code.

## Models

### 1. Keypoints

We initially experimented with bounding box detectors and segmentation models, but keypoints provided the best results and flexibility. The model outputs 6 pairs of XY coordinates (12 total), where 5 pairs are used for sagittal images, and 1 pair is designated for axial images. We incorporated the [corrected dataset labels](https://www.kaggle.com/code/brendanartley/lumbar-coordinate-dataset-code) shared by @brendanartley. This allowed us to use tilted crops, although the performance improvement was minimal.

The loss function was based on Euclidean distance, with masking applied to avoid propagating axial keypoint errors in sagittal images and vice versa. We also introduced position encoding to the backbone features before pooling to enhance the model’s performance.

### 2. Levels

For axial images, we developed a straightforward image model that predicts the corresponding level. Some metadata was concatenated with the backbone features to enhance the model's performance.

We observed occasional prediction inconsistencies (e.g., L5-S1 predicted next to L1-L2 within the same series). To address this, we redefined the task as a regression problem, predicting values between 0.0 and 1.0 at 0.25 intervals (e.g., 0.0 for L1-L2, 0.25 for L2-L3, etc.), using Mean Squared Error (MSE) as the loss function. This approach penalized predictions further from the true value more heavily.

### 3. Sequence

Leveraging anatomical symmetries, we trained one model with study-level-side inputs. 
- Input: For example, one row would be all patches in the L1-L2 right side, and another row all patches in the L4-L5 left side.
- Output: neural_foraminal_narrowing, subarticular_stenosis, spinal_canal_stenosis.

Details are explained further below.

#### Input

We used the keypoints and levels models to create patches, with each axial patch corresponding to one or more spinal levels. Since the levels model is a regressor, we applied a threshold with a tolerance range to capture additional context. For example, patches corresponding to L2-L3 (0.25 output) could span from 0.10 to 0.40.

To ensure consistency, we maintained uniform pixel spacing across each plane. We used 96x96 patches. Sagittal patches were offset at the top and bottom to prevent level leakage.

Some examples:


#### Pretraining: Patch model

Before training the sequence model, we pre-trained a patch model using individual patches with the same loss function (defined later). This step significantly improved the convergence of the sequence model.

#### Architecture

2D model + encoder-decoder transformer. 

The architecture combined a 2D image model with an encoder-decoder transformer. The best backbones we found were `regnetz_b16.ra3_in1k` and `hgnet_tiny.ssld_in1k`.

Metadata (modified XYZ world coordinates) was fed into the encoder, while image features were passed to the decoder. We added three learnable vectors to the decoder, each representing one disease classification (analogous to CLS tokens). Various pooling strategies were tested, including max/mean pooling and attention pooling.

#### Reshapable model

The model was designed to be flexible, allowing it to process either an entire study or individual level-sides. This adaptability enabled us to train the model at the level-side while evaluating it on the whole study.

What to do with spinal?

Since spinal stenosis affects the center rather than specific sides, we duplicated the label for both sides during training. At inference, we aggregated the learnable vectors from both sides (those added in the decoder) before passing them through the classification head. We tested multiple aggregation methods and ultimately chose max pooling.

#### Loss

We used the competition-specific loss function implemented in PyTorch. Although we explored several variations, such as a focal version of the competition loss, none yielded significantly better results.

## Further information

| description | backbone | local avg cv | public | private |
| --- | --- | --- | --- | --- |
| best public: ensemble 4 (dropped one) TTA 20 | regnet | 0.4027 | 0.3452 | 0.4121  | 
| best local: ensemble 5 TTA 40 | hgnet | 0.3883 | 0.3612 | 0.4215  | 
| best private: ensemble 5 TTA 16 | regnet | 0.4076 | 0.3472 | 0.4118  | 


The two selected submissions were the best local and the best in the public leaderboard. The best in the public leaderboard was the 3rd best in the private leaderboard.


## Code

https://github.com/claverru/RSNA-lumbar
