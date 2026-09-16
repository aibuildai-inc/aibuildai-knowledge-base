# 25th place solution

Competition: vesuvius-challenge-surface-detection
Rank: #25
Source: https://www.kaggle.com/c/vesuvius-challenge-surface-detection/writeups/25th-place-solution

I would like to express my gratitude to Kaggle and the organizers for hosting this competition. The challenge of extracting text from scrolls buried in volcanic ash is both fascinating and immensely difficult. By reviewing the code shared by the community, I have gained invaluable knowledge.

**Model Training**

Special thanks to @jirkaborovec for providing the foundational code (Surface nnUNet Training & Inference with 2x T4). Building upon this, I trained nnUNet models using the 3d_fullres configuration for 2000 epochs. Experiments were conducted with two distinct patch sizes: [128, 128, 128] and [224, 224, 224].
The training process leveraged both a local NVIDIA RTX 4090 GPU and the dual T4 GPUs provided by Kaggle. To identify the models with the optimal cross-validation (CV) scores, I employed an iterative training strategy, resuming training from checkpoints multiple times as needed.

**Post-processing**

Thanks to @ipythonx for sharing the inference code (Inference Vesuvius Surface 3D Detection). I adapted the post-processing function with slight parameter adjustments. However, due to Kaggle's submission limits, the final parameters selected may not represent the absolute optimum. The configuration used was as follows:

```python
final_mask = postprocess_topology(
    surface_prob=surface_prob,
    baseline_fg_mask=baseline_fg,
    low_threshold=0.25,
    high_threshold=0.8,
    z_radius=3,
    xy_radius=2,
    min_component_size=200
)
```

**Ensemble**

The final solution involved an ensemble of two models by performing a weighted average of their probability maps. The model trained with a patch size of [128, 128, 128] was assigned a weight of 0.7, while the model with a patch size of [224, 224, 224] received a weight of 0.3.
Notably, the [128, 128, 128] model achieved a higher public leaderboard score but underperformed on the private leaderboard. Conversely, the [224, 224, 224] model showed a lower public score but demonstrated superior performance privately. Consequently, fine-tuning these ensemble weights could potentially yield an even higher private score.

I joined the competition during the final stages. Given more time, I could have trained higher-performing models and incorporated additional architectures into the ensemble. This remains a promising avenue for further improving the score.
