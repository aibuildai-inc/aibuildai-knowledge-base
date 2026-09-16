# 18th Private LB Solution –> Silver

Competition: google-universal-image-embedding
Rank: #18
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359490

Our team would like to deeply appreciate the Kaggle staff and Google for hosting this exciting competition as well as everyone here who compete. My congratulations to all participants!

Special thanks to @ammarali32 for hard and effective teamwork. We really enjoyed this competition!

## Key points
- Custom dataset
- CLIP visual encoder trained on LAION-2B, finetune for 22 epochs
- Train the projection layer using only ArcFaceMarginLayer
- Use a single projection FC layer with the size of 64 (better than using multiple FC)
- TTA – upscale image 224 -> 240 helps to boost score by ~0.003-0.004
- Improve train time by embedding pre-extraction before the actual training starts. We found [this technique](https://www.kaggle.com/code/motono0223/guie-tensorflow-clip-distancelayer-train-dim64) lately and did not rely on it so much. 

## Dataset
- Google landmark recognition 2021 (7k classes, at most 35 images per class). Later we switched to a cleaned version of the GLR dataset, but, unfortunately, due to Out Of Memory exception happening on a regular basis, we were not able to test it completely
- Products10K (9691 classes, 141K images)
- [Open Food Facts dataset](https://www.kaggle.com/datasets/maximvlah/edaotkryt) (2k classes with at least 40 images per class) 
- [60,000+ Images of Cars](https://www.kaggle.com/datasets/prondeau/the-car-connection-picture-dataset) (322 classes, 12K images) 
- [Stanford Online Products](https://www.kaggle.com/datasets/liucong12601/stanford-online-products-dataset) (take only furniture, at least 10 images per class)
- [Landmarks (210)](https://www.kaggle.com/datasets/andreybeyn/qudata-gembed-landmarks-210) (210 classes, 10K images) 
- [Food 41](https://www.kaggle.com/datasets/kmader/food41) (101 classes, 3K images) 
- [Dishes 67](https://www.kaggle.com/datasets/kerrit/dishes-67) (67 classes, 670 images)
- [Food Recognition 2022](https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022) (500 classes, 12K images) 
- [Storefront](https://www.kaggle.com/datasets/kerrit/storefront-146) (141 classes, 4.5k images) 

## What did not work
- Strong augmentations while training
- Unsupervised learning with Contrastive loss
- Use images with zero padding for training
- [MET](https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings), art history datasets, 130K images 
- Distance layer, ElasticArcFace
- AdaptiveAveragePooling, GeM pooling, Multi-linear layers (neck)
- Use a bigger number of images per class / bigger number of classes for the GLR dataset
