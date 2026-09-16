# 33rd Place Silver Medal Solution

Competition: csiro-biomass
Rank: #33
Source: https://www.kaggle.com/c/csiro-biomass/writeups/33rd-place-silver-medal-solution

Honestly, I still cannot believe this. My first silver medal. 33rd out of 3000+ teams. I had to refresh the leaderboard three times to make sure it was real :)

First of all, thank you to CSIRO and everyone involved in organizing this competition. Predicting pasture biomass from images is such a real-world problem and I learned so much working on it.

A quick shoutout to the public notebooks and datasets that helped me along the way - the SigLIP model, the data split notebook, and the baseline approaches that many of us built upon. The Kaggle community is genuinely amazing.


## My Journey

I joined this competition about a month ago but if I am being honest, most of my serious work happened in the last 15 days. For the first few weeks I was just reading discussions, running public notebooks, trying to understand what works and what does not.

The public leaderboard was stuck around 0.72-0.73 for many people including me. I kept trying small things here and there but nothing was really moving the needle. Then I started thinking differently.

The main idea that changed everything for me was this: instead of using frozen DINO embeddings like everyone else, what if I actually retrain the DINO model from scratch on this specific dataset? And not just any DINO, but the huge version with 1.1 billion parameters.

This was risky because training such a massive model on only 357 images could easily overfit. But with proper regularization and early stopping, it worked. I went from 0.73 to 0.74 and jumped from 106th on public LB to 33rd on private LB.



## What I Actually Did

### The Model

I used vit_huge_plus_patch16_dinov3 as my backbone. This is a Vision Transformer with about 1.1 billion parameters pretrained with DINOv3 self-supervised learning.

Each pasture image is 2000x1000 pixels. I split it vertically into left half and right half, each 1000x1000. Both halves go through the same backbone (shared weights), then I concatenate the features together.

For fusing these features, I used something called Local Mamba Blocks. This is basically a lightweight attention mechanism with depthwise convolutions and gating:

```python
class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, 
                                 padding=kernel_size//2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.proj(x)
        return shortcut + self.drop(x)
```

The idea is simple: normalize, gate the information flow, apply local convolution, project back, and add residual. Two of these blocks stacked together gave me good feature fusion.

After fusion, I have three separate prediction heads for Green, Dead, and Clover. Then GDM and Total are computed from these (GDM = Green + Clover, Total = GDM + Dead). This respects the actual physical relationships between the targets.

### Training Details

Here is what worked for me:

- Image size: 512x512
- Batch size: 6
- 4-fold cross validation using StratifiedGroupKFold
- Learning rate: 1e-5 for backbone, 5e-4 for head
- Weight decay: 1e-2
- Dropout: 0.2
- Early stopping patience: 15 epochs
- Max epochs: 210 (but early stopping kicked in much earlier)
- Cosine annealing scheduler with 2 epoch warmup
- Mixed precision training (fp16)

For augmentation I kept it simple:
- Horizontal flip
- Vertical flip
- Random rotate 90
- Shift scale rotate
- Color jitter (mild)

The loss function is weighted to match the competition metric. I used Huber loss in log space to handle the wide range of biomass values and the outliers:

```python
weights = [0.1, 0.1, 0.1, 0.2, 0.5]  # Green, Dead, Clover, GDM, Total
loss = weighted_mean(HuberLoss(log1p(pred), log1p(target)))
```

### The Ensemble

DINO alone got me to around 0.72-0.73. To push further, I combined it with SigLIP.

SigLIP is a vision-language model. I extracted embeddings from images and computed similarity scores against agricultural text concepts like "bare soil", "dense pasture", "dead grass", "white clover" etc. These semantic features were then fed into a GBDT ensemble (HistGradientBoosting, GradientBoosting, CatBoost, LightGBM averaged together).

The final prediction is 75% DINO and 25% SigLIP for most targets. For Clover specifically, I use 100% DINO because SigLIP was not good at predicting it.

### Post-Processing

Small calibrations that helped:

```python
# Clover tends to overpredict
Dry_Clover_g = Dry_Clover_g * 0.8

# Dead material adjustments based on predicted range
if Dry_Dead_g > 20:
    Dry_Dead_g = Dry_Dead_g * 1.1
elif Dry_Dead_g < 10:
    Dry_Dead_g = Dry_Dead_g * 0.9

# Enforce mass balance
GDM_g = Dry_Green_g + Dry_Clover_g
Dry_Total_g = GDM_g + Dry_Dead_g
```



## Results

Cross-validation scores:

| Fold | R2 Score |
|------|----------|
| 0 | 0.82 |
| 1 | 0.85 |
| 2 | 0.81 |
| 3 | 0.84 |
| Mean | 0.83 |

Leaderboard:

| Stage | Score | Rank |
|-------|-------|------|
| Public LB | 0.73 | 105th |
| Private LB | 0.74 | 33rd |

The shakeup was real. Trusting CV over public LB paid off.



## What I Learned

1. Retraining beats frozen features - Fine-tuning DINO on this specific domain was the biggest gain
2. Respect the target relationships - GDM and Total are derived quantities, predictions should maintain this
3. Simple augmentation works - Heavy augmentation hurt more than helped on this small dataset
4. Regularization is essential - Dropout, weight decay, and early stopping prevented overfitting on 357 images
5. Trust your CV - Public LB was misleading, private LB matched my CV much better



## Code and Resources

I am sharing my complete solution:

**Training Code**: A clean single-cell notebook with the full DINO training pipeline. I ran this privately so the notebook does not show my actual training outputs, but the code is exactly what produced my 0.74 submission.

https://www.kaggle.com/code/ibrahimqasimi/csiro-biomass-33rd-rank-training-0-74-0-63

**Inference Code**: The ensemble inference that combines DINO and SigLIP. This is what generates the final submission.csv file.

https://www.kaggle.com/code/ibrahimqasimi/csiro-biomass-33rd-rank-inference-0-74-0-63

**Model Checkpoints**: The trained model weights (4 folds) that you can use directly for inference.

https://www.kaggle.com/datasets/ibrahimqasimi/dino-huge-retrain-checkpoints-zul

**GitHub Repository**: Complete code repository with training and inference scripts.

https://github.com/muhammadibrahim313/CSIRO-Image2Biomass-33rd-Place-Silver-Kaggle-Comp



Thank you for reading. This was my first silver medal and it means a lot to me. If you have any questions about the solution, feel free to ask in the comments.

PS: i  am now kaggle Competitions Expert :)  
Muhammad Ibrahim Qasmi
