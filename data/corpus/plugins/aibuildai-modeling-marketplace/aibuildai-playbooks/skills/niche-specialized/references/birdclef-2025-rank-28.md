# 28th Place Solution: SED with Segment-Based Voice Removal & Progressive Pseudo-Label Training

Competition: birdclef-2025
Rank: #28
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583377

First, I would like to thank the competition organizers for hosting this challenging and educational competition, and congratulations to all participants for their outstanding work! Special thanks to my teammate @ziyi777 for the collaborative effort throughout this competition!

## Solution Overview

Our final solution employs **Sound Event Detection (SED) models** with 5 different backbones trained through nearly identical pipelines, enhanced by two rounds of progressive pseudo-label fusion training and sophisticated ensemble techniques.

## Model Performance

|Backbone| Public | Private| Weight in Ensemble |
| --- | --- | --- | --- |
|efficientnetv2_b3| 0.878 | **0.893** | 0.2 |
|efficientnet_b0| 0.868 | **0.896** | 0.1 |
|efficientnetv2_s| 0.875 | **0.899** | 0.2 |
|seresnext26t_32x4d| 0.885 | **0.898** | 0.4 |
|eca_nfnet_l0 | 0.866 | **0.886** | 0.1 |
| **Weighted Ensemble** | **0.893** | **0.909** | - |

What's particularly encouraging is that our SED single models demonstrated remarkably consistent performance between public and private leaderboards, which we believe reflects the great generalization capability of our data processing and model training pipeline.

## Key Technical Innovations

### 1. Segment-Based Voice Removal Processing

Voice processing (especially for CSA files) was a challenging aspect of this competition. Through extensive experimentation, we developed a method that provides consistent improvements in single model Public LB performance while enhancing stability.

This algorithm is designed based on a key insight: **audio files are artificially concatenated, composed of multiple segments (bird call segments + human voice commentary segments), with silent intervals as transitions between segments**. The algorithm utilizes pre-computed voice detection results (Thanks to your excellent work! @kdmitrie) to identify human voice regions in the audio files.



#### Algorithm Steps:

**Silence-Based Segmentation**
- Detect silent intervals (amplitude < 0.001, duration > 0.3s) that serve as boundaries between concatenated segments
- Split the audio into independent segments based on these silence boundaries

**Voice Detection**
- Analyze each segment using pre-computed voice detection data to identify human voice regions
- Calculate the overlap between each segment and known voice regions to determine voice content percentage

**Segment Selection Strategy**
- Single segment: No processing (not artificially concatenated, no human commentary segments expected)
- Multiple segments: Keep only segments with the lowest voice content percentage

**Clean Audio Generation**
- Concatenate the remaining segments to form clean audio

Through this preprocessing, we can confidently **remove redundant human commentary while preserving environmental human voices that serve as background enhancement**.

### 2. RMS-Based Random Five-Second Sampling

After the voice removal process generates clean audio, this algorithm ensures we extract the highest quality 5-second segment for training. We randomly sample 3 five-second segments from the cleaned audio, calculate the RMS (Root Mean Square) energy for each segment, and select the segment with the highest energy.

This method maintains randomness in audio selection while increasing the probability of bird calls appearing within the random 5-second segments, which is crucial for the 5-second inference window used in evaluation.

#### Sampling Strategy Effectiveness (in order of performance):
1. **RMS-based Selection** (Our Choice): Highest performance by selecting segments with maximum energy
2. **Random 5-second Sampling**: Standard random selection baseline  
3. **First 5-second Sampling**: Simple but suboptimal approach

We focused our efforts on perfecting the 5-second window approach rather than exploring 10-second training windows due to the deadline, which needs more experiments in the future.

### 3. Progressive Pseudo-Label Fusion

We adopted and significantly improved the pseudo-labeling strategy from last year's 2nd place solution: randomly selected 5s audio segments from unlabeled test sets are added to training samples with a dynamic probability. Before mixing two audio signals, both waveforms' amplitudes are multiplied by random factors. The training sample's target vector (1.0 at primary and secondary species positions, zero elsewhere) is combined with pseudo-labels (vectors with prediction probabilities) by taking the maximum of both to form new target vectors.

#### Our Key Innovation: Progressive Pseudo-Label Mixup
Initially, we found that using fixed probabilities (35%, 45%) didn't improve new model performance and actually made it more unstable. We hypothesized that fixed-probability pseudo-label mixing might hinder the model's generalization to the soundscape domain.

**Our Intuition**: During early training stages, the model needs "harder" ground truth to learn fundamental knowledge, while in later stages, "softer" guidance can help the model gradually generalize to the soundscape space.

Based on this intuition, we chose to **linearly increase the mixing probability with epochs**, ultimately finding that 0.2-0.5 is an optimal range. This progressive approach allows the model to:
- Focus on learning from high-quality ground truth early on
- Gradually adapt to the distribution of unlabeled soundscape data
- Achieve better generalization without compromising fundamental learning

#### Critical Training Optimizations for Pseudo-Labeling:
- **Larger Batch Size**: We increased batch size to **128** during pseudo-label fusion training, along with proportionally scaling the learning rate. This larger batch size provides more stable gradient estimates when mixing real and pseudo-labeled data.
- **Backbone-Specific Training Epochs**: Through extensive experimentation, we determined the optimal training epochs for each backbone individually, then trained on full datasets before final LB validation. This careful epoch selection prevented both underfitting and overfitting (we're grateful this approach wasn't affected by leaderboard shake-up!).

### 4. Post-Processing

#### Temporal Smoothing:
We use temporal smoothing for the 1-minute soundscape predictions using carefully tuned weights:
```python
# For middle segments (i ∈ [1, N-2]) - our optimized 0.2-0.6-0.2 window:
new_pred[i] = 0.6 * pred[i] + 0.2 * pred[i-1] + 0.2 * pred[i+1]

# For boundary segments:
new_pred[0] = 0.8 * pred[0] + 0.2 * pred[1]
new_pred[-1] = 0.8 * pred[-1] + 0.2 * pred[-2]
```

This 0.2-0.6-0.2 weighting scheme was chosen after extensive experimentation and provides the optimal balance between temporal consistency and segment independence.



## Conclusion

Our solution demonstrates that **progressive pseudo-labeling combined with sophisticated audio preprocessing and ensemble techniques** can achieve strong performance in soundscape-based bird species identification. Key insights include the importance of voice removal preprocessing, RMS-based sampling strategies, and careful hyperparameter optimization for pseudo-label training.

Since we struggled with CNN models and many irrelevant details early in the competition, only finding the right direction for iterative improvement in the last month, we believe we could have achieved even better results with more time.

We hope our insights can be helpful to the community. Thank you again to all participants and organizers for making this such a rewarding learning experience!
