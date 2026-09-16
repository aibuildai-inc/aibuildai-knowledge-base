# Private 55th Solution and Lessons Learned

Competition: google-universal-image-embedding
Rank: #52
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359198

Thanks to Kaggle and Google for this amazing competition, @evilpsycho42 and I enjoyed our time a lot in this competition. It is a special competition in all the competitions I attended so far that need competitors to gather all domains of image datasets to get the best performing model possible.

Thanks to @dschettler8845 for initiating the external data thread and providing a list of links to help data, which saved us a lot of effort finding the data ourselves.

## Brief Solution Overview

### Data:
Datasets
- apparel: h&m
- landmark: google landmark competition
- package goods: product 10k, rp2k
- furniture: amazon product dataset
- storefronts: storefront-146 
- artwork: met
- dishes: ifood

Datasets sampling:
- Assign # of instances for each domain proportional to the ratio mentioned in this [post](https://ai.googleblog.com/2022/08/introducing-google-universal-image.html)
- Use different ids in the train and validation set, since we want to generate our models to unseen instances.


### Modeling
- backbone: clip 224/336 L + linear neck
- loss: ArcFace (scale = 10, margin = 0.3)
- training:
    - phase 1: freeze backbone + CosineAnnealing with warmup 0.1, lr=0.001, 100 epochs
    - phase 2: unfreeze backbone + CosineAnnealing with warmup  0.1, lr=1e-5, 20 epochs
- sampling strategy: id uniform sampling
- validation strategy: 
    - use different instances different from the train set
    - pick instance with > 1 images as query images, keep the rest as indexing images

### Results
- clip 224 L: cv 0.7949, public lb: 0.580, private lb: 0.598
- clip 336 L: cv 0.8074, public lb: 0.597, private lb: 0.614

### What doesn't work
- model fusion: 
    - train swin large 384 + clip together
    - train swin large 384 + clip separately, combine both backbone and train a new projection head
- image preprocessing\croppring with saliency segmentation

## Lessons Learned
After seeing the amazing solution from the 6th place [post](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/359161), we realized what we could have done better to deliver our final solution:
1. **We should have tried different backbones**: OpenCLIP ViT-H/14 224x224
2. **We should try to overfit our models first**. We used smaller lr and much fewer epochs to train the projection head while keeping the backbone frozen. Even though it is important to keep iteration fast, it is important not to be underfitting in our final solution!
