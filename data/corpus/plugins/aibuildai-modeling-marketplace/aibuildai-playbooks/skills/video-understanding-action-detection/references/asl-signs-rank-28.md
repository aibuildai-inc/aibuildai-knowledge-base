# 28th place solution - Using Arcface is not trivial.

Competition: asl-signs
Rank: #28
Source: https://www.kaggle.com/c/asl-signs/discussion/406630

Thanks to Kaggle and Google for hosting the competition :)
All of the work is done with @ydjoo12 

## Summary
We used 5-layers Transformer encoder with cross entropy and sub-class Arcface loss and ensemble 6-models trained with different combintation of cross entropy and sub-class Arcface loss.

## Data Preprocessing
- 129 keypoints are used. (21 for each hand, 40 for lip, 11 for pose, 16 for each eye, 4 for nose)
- Only use x and y.
- Pipeline: Augmentations, frame sampling, normalization, feature engineering and 0 imputations for NaN values.
- Frame sampling: When a length exceeds the maximum length, sampling was done at regular intervals up to the maximum length. It is better than center-crop in local CV and it minimized performance degradation while reducing maximum length.
    ```python
    L = len(xyz)
    if L > max_len:
        step = (L - 1) // (max_len - 1)
        indices = [i * step for i in range(max_len)]
        xyz = xyz[indices]
    L = len(xyz)
    ```
  
- Normalization: Standardization x and y independently.
- Feature Engineering
  - motion: current xy - future xy
  - hand joint distance
  - time reverse difference: xy - xy.flip(dims=[0]): +0.004 on CV


## Augmentations
- Flip pose: 
  - x_new = f(-(x - 0.5) + 0.5), where f is index change function.
  - +0.01 on Public LB
  - xy values are in [0, 1]. Shift to origin by subtracting 0.5 before flip, Back to original coordinates by adding 0.5
  - There was no performance improvement when not moving to the origin.
- Rotate pose: 

    ```python
    def rotate(xyz, theta):
        radian = np.radians(theta)
        mat = np.array(
            [[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]]
        )
        xyz[:, :, :2] = xyz[:, :, :2] - 0.5
        xyz_reshape = xyz.reshape(-1, 2)
        xyz_rotate = np.dot(xyz_reshape, mat).reshape(xyz.shape)
        return xyz_rotate[:, :, :2] + 0.5
    ```
  - angle between -13 degree ~ 13 degree
  - [Reference](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf)
- Interpolation (Up and Down)

    ```python
    xyz = F.interpolate(
            xyz, size=(V * C, resize), mode="bilinear", align_corners=False
    ).squeeze()
    ```
  - Up-sampling(Down-sampling) up to 125%(75%) of original length 


## Training
- 5-layers Transformer encoder with weighted cross entropy
  - Class weight is based on performance of the class.
  - Accuracy on LB and CV consistently improved until stacking 4~5 layers.
    - Public LBs => 1-layer(5 seeds): 0.714, 2-layers(5 seeds): 0.738, 3-layers(5 seeds): 0.748, 4-layers(5 seeds): 0.751
    - We add dropout layers after self-attention and fc layer based on the PyTorch official code.
- 5-layers Transformer encoder with weighted cross entropy and sub-class Arcface loss.
  - loss = cross entropy + 0.2 * subclass(K=3) Arcface
  - loss = 0.2 * cross entropy + subclass(K=3) Arcface
- Arcface
  - +0.01 on Public LB
  - Using arcface loss alone resulted in worse performance.
  - Arcface armed with cross entropy converges much faster and better than cross entropy alone.
  - Subclass K=3, margin=0.2, scale=32
- Scheduled Dropout
  - +0.002 on CV
  - Dropout rate on final [CLS] increased x2 after half of training epochs
- Label Smoothing
  - parameter: 0.2
  - +0.01 on CV
- Hyper parameters
  - Epochs: 140
  - Max lenght: 64
  - batch size: 64
  - embed dim: 256
  - num head: 4
  - num layers: 5
  - CosineAnnealingWarmRestarts w/ lr 1e-3 and AdamW


## Ensemble
- 2 different seeds of Transformer with weighted cross entropy
  - Single model LB: 0.75
- 2 different seeds of weighted cross entropy + 0.2 * subclass(K=3) Arcface
  - Inference: Weighted ensemble of cross entropy and Arcface head.
  - Single model LB: 0.76
- 2 different seeds of 0.2 * cross entropy + subclass(K=3) Arcface
  - Inference: Weighted ensemble of cross entropy and Arcface head.
  - Single model LB: 0.75
- 6 models ensemble Public LB: 0.779
- All models are fp16
- Total Size: 20Mb
- Latency: 60ms/sample


## Working on CV but not included in final submission
- TTA
  - +0.000x on CV
  - Sumbission Scoring Error. It might be a memory issue. 
- Angle between bones of each arm
  - 0.000x on 1 fold. We couldn't fully validate it due to time limits.

## Not working
- GCN embedding layer instead of Linear
- Stacking Spatial Attention & Temporal Conv. blocks
- Distance between pose keypoints
- Removing outlier and Retraining
  - We used anlge between learned Arcface subclass vector
  - About 5% (~4000 samples) are removed
- Knowledge distillation with bigger Transformer
- Stochastic Weight Average
- Using all [CLS] in every layer
- Average all tokens instead of [CLS] token
- Stacking with MLP as meta-learner
