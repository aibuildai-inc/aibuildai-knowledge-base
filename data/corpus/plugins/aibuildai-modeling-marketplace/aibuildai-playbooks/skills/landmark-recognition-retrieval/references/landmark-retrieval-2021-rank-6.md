# 6th place solution

Competition: landmark-retrieval-2021
Rank: #6
Source: https://www.kaggle.com/c/landmark-retrieval-2021/discussion/276632

Fisrst of all, many thanks to Kaggle and the hosts for hosting such an interesting competition, and congratulations to all the winners. And a special thanks to all my teammates @takuok and @tascj0 .

# Data definition
- train_clean:  there are 1.6 million training images and 81k classes.
- train3.2m: there are 3.2 million images belonging to the 81k classes in train_clean.
- train4.1m: all train data in GLDv2, that is 4.1 million images.
- train4.9m: all train data in GLDv2 and index images of 2019 competition.

# Tips in Retrieval

We used reranking techniques proposed by smlyaka team of the 2019 GLR champion. 
**Github**: https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution
**Paper**: https://arxiv.org/abs/2003.11211
There are two steps, sort-step and insert-step. First, landmark_id and score of index set and query set are determined using train4.9m. Then, in the sort-step, retrieved index results are sorted based on their label similarity to the query images. In the insert-step, we added samples that are not retrieved by their image similarity but have same landmark_id as the query image.
This improved the base score from public/private 0.31324/0.32614 to 0.39905/0.43219.

## Model

The models we used are almost same as Recognition ones.
We concatenate embedding vectors of our models shown below, then use them to calculate similarities.


### takuoko part

I used almost the same architecture that the last year 1st team used. `ViT backbone -> Dense(1024) -> ArcFace` with class weight. I chose ViT backbone because it works well on [FGVC tasks](https://paperswithcode.com/task/fine-grained-image-classification). I had thought about using FGVC tricks to get local and global features, but I didn't have enough time.
I also suffered from the phenomenon of loss becoming nan when learning with fp16. I got around this by the grad_clip and calculating Attention layer using fp32.

Base settings
- image size: 384
- optimizer: AdamW
- scheduler: cosine with warmup
- batch size: 16*2GPU
- use fp16

**BeiT-large: 2epochs@(train_clean) -> 8epochs@(train4.1m)**
**Swin-base: 30epochs@(train_clean) -> 6epochs@(train4.1m)**
Sadly, the 2nd part of my swin didn’t work well.

### tascj part
I used augmentations, loss (sub-center ArcFace) and training strategies following last year’s top solutions.

**b4: 10epochs@(train_clean, 512) -> 6epochs@(train4.9m, 512)**
**b6: 10epochs@(train_clean, 256) -> 6epochs@(train4.9m, 512) -> 2epochs@(train4.9m, 640)**
**xcit_small_24_p16_384: 6epochs@(train_clean, 384) -> 6epochs@(train4.9m, 384)**


### inoichan part
Architecture is sub-center ArcFace with dynamic margins which was used in last year’s 3rd-place team. Backbone -> Dense(512) -> sub-center Arc. Training strategy is as following:

##### efficientnet v2m
**1st**
data: train_clean
image size: 256 crop from 300
epoch: 15
batch size: 64
scheduler: warmup

**2nd**
data: train4.1m
image size: 512 crop from 600
epoch: 15
batch size: 8 x 16 steps w/ freeze batch norm
scheduler: cosine annealing

**3rd**
data: train4.1m
image size: 640 crop from 720
epoch: 3
batch size: 4 x 64 steps w/ freeze batch norm
scheduler: cosine annealing


##### efficientnet b5
**1st**
data: train4.1m
image size: 256
epoch: 10
batch size: 32 x 8 steps w/ freeze batch norm
scheduler: cosine annealing

**2nd**
data: train3.2m
image size: 512
epoch: 7
batch size: 16 x 16 steps w/ freeze batch norm
scheduler: cosine annealing

**3rd**
data: train4.1m
image size: 512
epoch: 9 (freeze backbone in first two epoch, then unfreeze)
batch size: 16 x 16 steps w/ freeze batch norm
scheduler: cosine annealing


# Acknowledge
takuoko is a member of Z by HP & NVIDIA Data Science Global Ambassadors.
Special Thanks to Z by HP & NVIDIA for sponsoring me a Z8G4 Workstation with dual RTX6000 GPU and a ZBook with RTX5000 GPU.
This competition has the big dataset.
So I tried pytorch's DDP parallel training on my dual RTX6000 GPUs and it helped a lot.
