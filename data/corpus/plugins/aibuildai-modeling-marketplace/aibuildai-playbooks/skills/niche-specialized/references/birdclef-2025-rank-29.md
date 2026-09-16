# 29th Place Solution for the BirdCLEF+ 2025 Competition

Competition: birdclef-2025
Rank: #29
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583387

## Our Top Solution

Thanks to the organizers for hosting this interesting competition, and thanks to past Kagglers whose work for previous BirdCLEF competitions gave us great inspiration. This was a challenging competition because the public test and private test distributions didn’t seem to correlate, so it was tough to know if good validation scores would transfer from our local validation to the private test. Also, thanks to my teammates @siavrez and @kainsama for their innovative work and effort for this competition!

In any case, our final solution for this competition consists of these parts: 

### Backbone Models

* EfficientNet-B3 (single-channel stem, ImageNet init)
* ResNeSt-50 (wider stem 64→128, ImageNet init)
* Nfnet-l0 (single-channel input, tuned layer norms)

### Data Processing

* Train on 256*313(2x,3x,4x,6x) segments
* Standardize with global mean/std from ~10k samples
* On-the-fly augmentations: time shift ±10 frames, freq shift ±5 bins; SpecAugment masks (time ≤ 15, freq ≤ 6); 50 % mixup


### Loss Function

* Multi-class focal loss (γ = 1.5, α = 0.25) to focus on hard/rare classes

### Model Training

* AdamW (lr = 2 × 10⁻⁴, weight_decay = 1 × 10⁻⁵), cosine decay with 3-epoch warmup
* Training on 10-15-20-30 second segments, and infer on 5-second segments
* Average top 5 checkpoints per model based on validation LRAP

### Ensembling

* In-model: average logits of top 5 checkpoints
* Cross-model blend (tuned on hold-out): EfficientNet 0.48, ResNeSt 0.32, ConvNeXt 0.20

### Other Things

* Trained on all data for the final solution
* Used ONNX to improve the runtime performance.

Our top solution got a score of 0.902 on the public leaderboard and 0.906 on the private leaderboard.

Link to the [inference notebook](https://www.kaggle.com/code/siavrez/another-blend-combo-shift-rare?scriptVersionId=243860856)
