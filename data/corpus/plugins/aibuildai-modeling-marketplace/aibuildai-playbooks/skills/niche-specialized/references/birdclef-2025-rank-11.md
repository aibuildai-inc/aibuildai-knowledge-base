# 11th solution

Competition: birdclef-2025
Rank: #11
Source: https://www.kaggle.com/c/birdclef-2025/discussion/583384

Thank you to the organizers for organizing such an interesting competition, and also thank the excellent plans from previous sessions for inspiring me. Also, congratulations to all the top winners. Next, I will introduce my solution.

# 1、Training Data

- The 206 categories have been expanded to 316

(1) 25 year competition data: train_audio and train_soundscapes

(2) select 110 categories (categories with a sample size of less than 10) from the data of previous competitions

This additional category is mainly for building local cv. Unfortunately, this cv strategy is ineffective. However, this mixed training improved my lb, so I maintained this operation

- The maximum sample size for each category is 500. Categories smaller than 10 will be upsampled
- Only remove human voices from part of the CAS data

# 2、**Model Architecture**

Sed models opened by [2nd place solution of 2023](https://www.kaggle.com/competitions/birdclef-2023/discussion/412707). I added the code related to pseudo-labels on this basis

backbones are：

- tf_efficientnetv2_b3
- tf_efficientnetv2_s

All of them are trained on 10sec clip.

# 3、Loss function

Using ce loss, compared with BCE loss, there is a qualitative improvement (0.83→0.88) in multiple models.

# 4、**Mel Spectrogram Parameters**

```
{'sample_rate': 32000, 'n_mels': 256, 'image_size': 300, 'f_min': 90, 'f_max': 14000, 'n_fft': 1536, 'normalized': True, 'hop_length': 535}

{'sample_rate': 32000, 'n_mels': 256, 'image_size': 300, 'f_min': 50, 'f_max': 14000, 'n_fft': 1024, 'normalized': True, 'hop_length': 535}
```

# 5、pseudo label

- Select high-quality pseudo-labels based on the entropy-based screening strategy mentioned in [rank 10  of last year 24](https://www.kaggle.com/competitions/birdclef-2024/discussion/511596)

```markdown
# 示例代码
import numpy as np
epsilon = 1e-12

pre_probs = sub_df[columns].values
print(pre_probs.shape)

probs = pre_probs.copy()
entropies = -np.sum(probs * np.log(probs + epsilon), axis=1)
print(entropies.shape)

# 筛选前 20% 的伪标签
top_10_indices = np.argsort(entropies)[:int(len(entropies) * 0.2)]
top_10_pseudo_probs = probs[top_10_indices]
print(top_10_pseudo_probs.shape)

# 对于每个类别的伪标签，将低于前 92% 的标签值设置为 0
for i in range(top_10_pseudo_probs.shape[1]):
    class_probs = top_10_pseudo_probs[:, i]
    threshold = np.percentile(class_probs, 92)
    top_10_pseudo_probs[class_probs < threshold, i] = 0
print(top_10_pseudo_probs.shape)

```

- The real dataset and the pseudo-label dataset are concatenated into the model, and the loss of the pl part is down weighted

batch_size = 96，pl_batch_size = 16

# 6、**Ensemble and Post-processing**

- ensemble model

5 ✖️ v2b3 + 1✖️v2s

Public Score: 0.920
Private Score: 0.919


- post precessing

The post-processing is the same as [the rank 6 in 2024](https://www.kaggle.com/competitions/birdclef-2024/discussion/511527)，An increase of approximately 0.001

```
def smooth_array_general(array, w=[0.1, 0.2, 0.4, 0.2, 0.1]):
    smoothed_array = np.zeros_like(array)
    timesteps = array.shape[0]
    radius = len(w) // 2

    for t in range(timesteps):
        for i, weight in enumerate(w):
            index = t - radius + i
            if index < 0:
                smoothed_array[t] += array[0] * weight
            elif index >= timesteps:
                smoothed_array[t] += array[-1] * weight
            else:
                smoothed_array[t] += array[index] * weight
    for c in range(array.shape[1]):
        smoothed_array[:, c] = smoothed_array[:, c] * 0.8 + smoothed_array[:, c].mean(keepdims=True) * 0.2
    return smoothed_array
```

# 7、not work

- CNN
- rms sample
- remove all human voices
