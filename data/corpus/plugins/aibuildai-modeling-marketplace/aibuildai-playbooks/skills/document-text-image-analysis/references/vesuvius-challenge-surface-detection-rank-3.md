# Quick Preview of the 1st place solution

Competition: vesuvius-challenge-surface-detection
Rank: #3
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/quick-preview-of-the-3rd-place

Our formal write up will be posted (by Paul) soon.

The most excited thing today is that we successfully select the highest private LB solution. 



For quick preview, the following is the model training and ensemble part.

First, we would like to appreciate [Jirka](https://www.kaggle.com/jirkaborovec)'s work! His [notebook](https://www.kaggle.com/code/jirkaborovec/surface-train-inference-3d-segm-gpu-augment) served as the starting point of our training pipeline.

### Single Model Strategy

- We used all available data for training and built our baseline nnU-Net model (Model 1) with the following settings:

```
patch size: 128
batch size: 2
epochs: 4000
```

- We then fine-tuned Model 1 with larger patch sizes of 192 and 256, training for 250 epochs each.

- Due to the extension of the competition, we trained additional models from scratch (for 4000 epochs) with patch sizes of 160, 192, and 224 to increase model diversity. Among these, the 192-patch model was included in our final ensemble.

### Ensemble Strategy

We prepared two sets of 4-model ensembles as our final submissions:

- **Set 1:** Baseline 128-patch Model 1 (weight: 0.12) + fine-tuned 192-patch model (0.28) + fine-tuned 256-patch model at 100 epochs (0.18) + fine-tuned 256-patch model at 250 epochs (0.42).
- **Set 2:** Baseline 128-patch Model 1 (weight: 0.12) + fine-tuned 192-patch model (0.28) + fine-tuned 256-patch model at 250 epochs (0.18) + from-scratch 192-patch model at 4000 epochs (0.42).

For each ensemble, we fused the post-softmax probabilities using the assigned model weights and applied different thresholds and post processing methods.

| Ensemble | Public LB | Private LB | Threshold |
|----------|-----------|------------|-----------|
| Set 1    | 0.613     | 0.620      | 0.20      |
| Set 2    | 0.606     | 0.627      | 0.26      |
