# 11th Place Solution: Supervised Contrastive Pretraining + Query and Database expansion

Competition: landmark-retrieval-2021
Rank: #11
Source: https://www.kaggle.com/c/landmark-retrieval-2021/discussion/276338

Thanks to Google and Kaggle for organizing this interesting competition! 
Congratulations to the winners!

It was the first large-scale competition for both teams @ofitserovlad, @evgenysidorov, @joven1997 and we spent a lot of time on the experiments and came up with a strong pipeline pretty close to the end of the competition. Even though we finished just short of the gold medal we are still more than satisfied with the results and enjoyed the competition immensely.

# Solution Overview


# 1. Supervised Contrastive Pretraining
### Models
Efficientnetv2_m

### Validation 
mAP calculated with Index (100k sampled) and Test (1k) data from google’s repository. CV and LB had good correlation with LB being almost always 0.015-0.02 lower than CV (ex. CV 0.380 -> LB 0.360)

### Training Pipeline 


Team Fat Cat finalized a pretty strong 3 stage training process a bit too late (1.5 week before the end) into the competition:
Artificially Sampled Supervised Contrastive Pretraining with class balanced temperature on 448 resolution and clean dataset 
Finetuning with AdaCos with class balanced Focal Loss with 512 resolution on clean dataset 
Same as stage 2 but with 604 resolution and full dataset 

### Artificially Sampled Supervised Contrastive Pretraining with class balanced temperature 
Pretraining part was inspired by this [paper](https://arxiv.org/abs/2004.11362). At this stage the model should learn easy image features and be more generalizable before finetuning stage. The premise of the paper is pretty similar to SimCLR, which takes 2 augmented (color jitter, random flip) random cuts of the image and makes them close in the cosine space. The extension of Supervised Contrastive paper is to make cuts of the image of the same class in the batch also close together. This paper also normalizes the image embeddings before feeding them into the nonlinear projection head. Outputs of the projection head are then used for supervised contrastive loss. 

The main problem with implementation of the original paper was that they used enormous batch size (8192) and only 1000 classes and by doing so are somewhat guaranteed to have multiple instances of the same class in the batch. When I tried that kind of approach with a batch of 256 cv was at ~0.180, clearly indicating that there was very little probability of the same class occurring in the batch. 

1.5 weeks before the end I realized that I can artificially sample cuts of the different images of the same class into 1 batch. I shuffled my dataset in such a way that each 4 consecutive images were augmented cuts of the same class. This kind of pretraining resulted in cv mAP of .273 (and was much faster than cutting 2 parts of the same image), after 10 epochs of pretraining. Graphs showed that given more time it could reach .300. For pretraining I used SGD with cosine warmup and annealing. 

Additionally, individual temperature for each class was used. It was derived using class-balanced strategy and omitting final normalization (i.e sum of the weights equals to the number of classes). 

### Finetuning 
For finetuning Adaptive Cosine loss was used. It is an extension of AcrFace and was also used by the previous year’s winner. Supervised Contrastive pretraining was a crucial step for faster convergence during finetuning. Without pretraining after 1 epoch training on clean data and 512 resolution CV mAP was at .210 and with pretraining it started at 0.31 and reached 0.35 after 10 epochs with SGD and cosine scheduling. However, with better hyperparameters it potentially could have climbed even higher. Then, the resolution was increased to 604 and finetuned on the full dataset. 

At the very end of the competition we added additional weight to landmarks from Asia, Africa, and Oceania to try to match the distribution of the test data which was described by Google's paper. It helped to bridge the gap between LB and CV. 

# 2. Encoder + GEM + ArcHead
 
### Networks:
EfficientnetB6, 704x704
EfficientnetB7, 664x664
Seresnext101_32x4d, 512x512

### Training procedure:
ArcFace loss based on Focal Loss with various modifications was used to train the models. 
We quickly realized that big models with big resolution give improvements up to 0.03-0.05.
Training big models was optimized by using FP16 precision, RandomCrop augmentations to reduce training image size (for example, from 704 to 664) and doing all the augmentations on GPU with Kornia.
One model training can be divided into two steps:
Training on GLDv2 cleaned
Fine tuning on GLDv2 full dataset with smooth labeling and doubled weights for cleaned samples to reduce effect of the noisiness 
 
### 3. Post Processing
Two stage post processing procedure was suggested by Fat Cat. It consisted of query expansion and database expansion:


Query expansion - for each sample from test set we find top2 samples from index set by cosine similarity and generate new embedding for test sample by normalizing sum of weighted by cosine similarity score top2 index samples and test sample
Database expansion - for each test and index sample we find top16 samples from GLDv2 train set and based on their classes define the class of the test/index sample. Using this classes we re-rank top100 list of index samples for each test sample that we get after qe expansion. We put on top of the new list samples from the top100 list that have the same class as the test sample in the same order. Then we put samples from the index set that have the same class as the test sample but weren't in the top100 list before sorted by cosine similarity score, to the end of the new list we put all samples from the top100 list that have different class than the test sample. Finally, top100 samples from the new list are taken.
This post processing strategy improved our LB score by around 0.07.
 
### Ensemble
For all the post processing procedures we used faiss to speed it up using GPU instead of CPU. For testing the ensemble we used Biba & Boba’s Seresnext101_32x4d model (0.418 LB) and Fat Cat’s model (0.425 LB). Our team explored two different ways of merging models.
- “Majority voting” - from each model in the ensemble top100 samples was found independently and then all the image ids were sorted based on the sum of the positions that this ids has in each model’s top100 list. Finally, the top 100 indices from the sorted list were taken. (0.433 on LB)
- “Concat” - concatenating normalized model’s embeddings. Before concat each embedding was multiplied by the weight given to its model. When all embeddings are concatenated we normalize the new big one. To new big embedding we applied two staged post processing described above. (0.455 on LB)
 
The final ensemble were made with:
Two EffientnetV2 M models trained with supervised contrastive pretraining + AdaCos
Seresnext101_32x4d and EfficientnetB6 “Encoder + GEM + ArcHead” models

 
### Final Thoughts 
It was both teams' first large-scale kaggle competition so a lot of different mistakes were made along the way. After all experiments here are the ingredients of what we believe could be very strong training pipeline
- Better cropping [paper](https://arxiv.org/abs/1906.06423)
- Artificially sampled Supervised contrastive pretraining with 8 cuts of different images (here we used 4) with 448 validation resolution and 20 epochs on clean data
- Finetuning with ArcFace 512 validation resolution and class balanced focal loss for 20 epoch with sgd without schedulers on clean data
- Same as previous previous step but with 736 and total data
- Ensemble by concatenation 
- Post processing: qe + db expansion
