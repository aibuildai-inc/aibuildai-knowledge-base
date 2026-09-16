# DOLG Army: 7th Place Solution Summary

Competition: happy-whale-and-dolphin
Rank: #7
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/320026

Hi all, 

Congratulations to all the winners. And thank you to kaggle host and all participants for this exciting competition.It was as always very tough last week with huge LB changing everybody , fortunately for us , our ideas worked and we ended on right side of the private LB.

Congratulations to @ks2019 for becoming Kaggle Competitions GM. Thanks a lot to my teammates @nischaydnk @tanulsingh077 @navjotbansal. Such a great team effort 🤜.

Our solution consists of the following major components:
1. Data Recipe
2. Modelling
3. Progressive Psuedo Labelling
4. PostProcessing
5. Ensemble

All our models are trained on TPUs using more or less the same tensorflow pipeline that KS shared in the beginning of the competition.

## Data Recipe
There were different datasets available publically, which we thought could help us in diversity. Inspired from @thedrcat 's solution in Chaii, we decided to make our own data recipe. We used the following combination of datasets to train our models:
- Fullbody Annotations
- Fullbody Annotations + Backfins concatenated horizontally  
- Fullbody Annotations + Original Images concatenated horizontally

Fullbody Annotations
<a href="https://ibb.co/LPtzkGQ">[fullbody]</a><br /><a target='_blank' href='https://imgbb.com/'></a><br/>

Fullbody Annotations + Backfins concatenated horizontally 
<a href="https://ibb.co/kH7CW6G">[fullbody-backfin]</a><br /><a target='_blank' href='https://imgbb.com/'></a><br/>

Fullbody Annotations + Original Images concatenated horizontally
<a href="https://ibb.co/mz25mhN">[fullbody-original]</a><br /><a target='_blank' href='https://imgbb.com/'></a><br/>

We used simple augmentations:
- horizontal flip
- random pixel based augmentation (brightness, contrast, HSV)
- cutout

## Modelling
We used a combination of DOLG (with EFFNet backbone) and normal EFFNets with CurricularFace loss. We used @christofhenkel 's implementation of DOLG model, [implemented in PyTorch](https://www.kaggle.com/competitions/landmark-retrieval-2021/discussion/277099) which we ported to TensorFlow because it worked better than the public implementation in this competition. We also used multiple heads in all the models: one for species classification and another for individual classification. Species classification head was trained with normal softmax loss while the individual classification head was trained with CurricularFace loss.

Models in our final submission:
- DOLG B5/B6/B7, image_sizes: (786, 786x2), (896, 896x2), dataset: Fullbody Annotations + Backfins
- DOLG B6/B7, image_sizes: (896, 896x2), dataset: Fullbody Annotations + Original Images
- DOLG B5/B6/B7, image_sizes: (1024, 1024), dataset: Fullbody Annotations
- EFFNet B5/B6/B7, image_sizes: (786, 786x2), (896, 896x2), dataset: Fullbody Annotations + Backfins
- EFFNet B5/B6/B7, image_sizes: (1024, 1024), dataset: Fullbody Annotations

All the models were trained on psuedo labelled data from our best ensemble. During inference we also use hflip as TTA.

## PostProcessing
We used a second stage model for getting better confidence scores which is less susceptible to threshold changes. The idea was to use predictions/embeddings/features to get a better confidence score than just using nearest distances. After candidate generation (top100 neighbors based on knn distances) we engineered the following features for our 2nd stage model (features were generated from the models stated above):
- species probabilites for each image_id
- top3 nearest distances for each (image_id, unique individual_id) pair present in the candidates
- distance of each image_id from centroid of each unique individual_id present in the candidates
- rank of each unique individual_id present in the candidates
- sum of top3 neighbor distances 
- OOF predictions

We trained a 5 folds XGB and LightGBM models on the above features and used a ensemble of their predicitons as final confidence scores.

## Ensemble
We used simple weighted voting approach using the confidence scores obtained from above where the weights were optimized using 5 fold OOFs.

## Acknowledgements
- Thanks a lot to @jpbremer for sharing the datasets (The real hero 🔥)
- Special mention to Google TPU Research Program (https://sites.research.google/trc/) which helped us in training bigger models in the last few days of the competition by supporting us with V3 TPUs on GCP.
