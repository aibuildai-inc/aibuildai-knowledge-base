# 25th solution

Competition: google-universal-image-embedding
Rank: #25
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359410

Thanks for organizers hosting a nice competition.

Initially I was working on this competition with text-image contrastive method and trying dimension reduction technique like UMAP and also trying embedding each category(fashion/package/landmark...) on different discrete spaces, but these approach were not good.

The approach of [motono0223's baseline](https://www.kaggle.com/code/motono0223/guie-clip-tensorflow-train-example) was best for me. freezed CLIP + Arcface head.

## Downsampling for the number of classes
* 64D embedding  space is not large so I tuned the number of classes for training.
* Training 4000 classes from Products10k/GLR was the best for me.
[image]

## Hard to create good CV-LB correlation
* I configured 6 different retrieval tasks, Products10k/GLR/Stanford Online Products/DeepFashion/MET/Food-101/ObjectNet but could not find clear CV-LB correlation.
* When only training with Products10k/GLR, Stanford Online Products(Cabinet/Sofa/Chair) retrieval setting was relatively correlated to LB. But it was still not perfect one.
[image]
## Freeze CLIP
* Unfreeze CLIP finetune does not worked for me. Lowering learning rate got worked but freeze & high learning rate was better.
* Fintuning other backbones, like Imagenet pretrained models, were not good.
* Adding more transformer layers on freezed CLIP was not good.

## My code for this competition
* https://github.com/Fkaneko/kaggle_google_universal_image_embedding
