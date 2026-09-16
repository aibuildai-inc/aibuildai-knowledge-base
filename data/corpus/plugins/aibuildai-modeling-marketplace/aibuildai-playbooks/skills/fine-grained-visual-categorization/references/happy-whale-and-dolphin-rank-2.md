# 2nd place solution

Competition: happy-whale-and-dolphin
Rank: #2
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320502

Thank you to the competition host, Kaggle and all participants. And congrats to the winners.
Many thanks to @jpbremer. We used the public fullbody dataset and annotation as our main dataset.  Without it we could not achieve a high score in a short period of time.

# Single model
## Dataset
Two different datasets were used. 
- fullbody dataset (yolov5): [this dataset](https://www.kaggle.com/competitions/happy-whale-and-dolphin/discussion/311184) cropped by yolov5. 
- fullbody dataset (yolox): the above annotations were used for GT and trained yolox.

All models were trained using the fullbody dataset (yolov5) before the last day of the contest. Finally, a few models were fine-tuned a few epochs using the fullbody dataset (yolox).

## Model
efficientnet_l2 worked the best in validation

### Training Recipe
- backbone = tf_efficientnet_l2_ns
- img_size = 768
- loss = Arcface with adaptive margin
- augmentation = Horizontal flip, RandAugment
- optimizer = SGD
- scheduler = CosineDecay with warmup
- batch_size = 16 per GPU
- n_epoch = 20

RandAugment improved the validation score quite a bit.

## Pseudo labeling

This is the key to getting a high score on the leaderboard.

We use FC layer prediction (`(logits * scale).softmax(-1)`) of trained models to generate pseudo labels. The confidence threshold was set to 0.8.
Every pseudo labeling round was trained from imagenet pretrained weights.

Following are the leaderboard scores of each round. We used flip testing starting round3.

Pseudo label rounds: backbone = efficientnet_l2
| Pseudo label round     | Dataset           | Public score | Private score |
|------------------------|-------------------|--------------|---------------|
| initial model (round1) | fullbody (yolov5) | 0.846        | 0.812         |
| round2                 | fullbody (yolov5) | 0.875        | 0.849         |
| round3                 | fullbody (yolov5) | 0.885        | 0.860         |
| round4                 | fullbody (yolov5) | 0.889        | 0.862         |
| round5                 | fullbody (yolov5) | 0.887        | 0.863         |
| round5 (fine-tuning)   | fullbody (yolox)  | 0.891        | 0.870         |

## Make submission
The normal image retrieval method. Calculate cos similarity with train dataset and get top5 ids.

A fixed cosine similarity of 0.5 was used to insert “new_individual”. There is nothing special post-processing, such as species-specific thresholds.

# Last ensemble
The final submission is a 4-model ensemble of efficientnet_l2 belows. (public=0.897 / private=0.872)

Ensemble models: all backbone = efficientnet_l2
| Pseudo label round | Image size   | Dataset                                                     |
|--------------------|--------------|-------------------------------------------------------------|
| round3             | (1024, 1024) | fullbody (yolov5)                                           |
| round4             | (768, 768)   | fullbody (yolov5)                                           |
| round5             | (768, 768)   | fullbody (yolox) fine-tuned                                 |
| round5             | (768x2, 768) | vertical concatenated original dataset and fullbody (yolox) |


# Performance tips
## Gradient checkpointing

To train efficientnet_l2 on RTX3090, gradient checkpoint is a must. With gradient checkpointing and mixed precision, we could train the network with batch_size 16 on a single RTX3090. Without it, even batch size 2 gives OOM.

## PyTorch build

We found that different PyTorch builds can greatly affect training throughput. We compared the official release and the build by NVIDIA.

| docker image                                 | cudnn_benchmark   |   training throughput (imgs/sec) |
|:---------------------------------------------|:------------------|----------------------:|
| pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel | False             |                   2.5 |
| pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel | True              |                   3.4 |
| nvcr.io/nvidia/pytorch:22.02-py3             | False             |                   4   |
| nvcr.io/nvidia/pytorch:22.02-py3             | True              |                   4.1 |

The build from NVIDIA is 20% faster. Probably because of the updated CUDA/CuDNN version.

# Acknowledge
takuoko is a member of Z by HP Data Science Global Ambassadors. Special Thanks to Z by HP for sponsoring me a Z8G4 Workstation with dual A6000 GPU and a ZBook with RTX5000 GPU.
