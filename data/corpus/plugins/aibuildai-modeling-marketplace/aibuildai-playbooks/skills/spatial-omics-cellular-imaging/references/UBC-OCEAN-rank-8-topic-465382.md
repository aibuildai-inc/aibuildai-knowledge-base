# [8th Place Solution] Understanding Data Before Designing Methods

Competition: UBC-OCEAN
Rank: #8
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465382

# Acknowledgments  
   
I would like to express my gratitude to Kaggle, the organizers, and other participants in the community. I have learned a great deal from this competition and hope that it will promote the advancement of research in MIL，Classification and outliers detection of Human tissues, and the study of women's health.

# Introduction  
   
The first and most crucial step in this competition is to familiarize oneself with the feature of Whole Slide Images (WSI) and Tissue Microarray (TMA) images. A simple visual examination reveals that **the features of TMA and WSI are presented at vastly different scales.** TMA features are at the cellular level, while WSI features are several to tens of times larger than cell clusters. Hence, from the onset of the competition, I decided to tackle WSI and TMA with two completely distinct approaches.  

## Key points for the global approach:  

1.  **Address WSI and TMA separately** .
2.  **Align the magnification of TMA and WSI** to enable the reuse of TMA training code for optimizing the WSI feature extractor.  
   - WSI images are 20x magnified, while TMA images are 40x. Thus, TMA images need to be downsampled by a factor of 2 to align with the physical scale of WSI, which can also be observed with a visual inspection of the training data.  
   
## Architecture of the solution



## TMA Approach  
   
### Summary of Key Points:  
1. **Train with patches tiled with official masks** (most important globally).  
2. Train the model with **healthy and death patches** to predict some outliers.  
3. Employ **arcface** to retrieve some outliers and part of the 5-class classification.  
4. Heavy ensemble of 6 classification models to predict samples that arcface retrieval did not cover.  
   
### Inference:  

1. Use arcface retrieval with the training set's TMA as templates. Output 5-class results for samples with a close cosine distance, and consider those with a far cosine distance as outliers. Samples that are uncertain in the arcface stage are left for the subsequent phase.  
   - Models: 5-fold effv2s + 5-fold convnext small with dynamic margin, subcenter=3.  
   - Top1 threshold at 0.05, Other threshold at 0.2.  
  
2. The 6-class model uses 2-fold effv2s + 5-fold effv2l + 4-fold convnext small + 3-fold convnext large.  
 
| Model | Resolution |
| --- | --- |
| Effv2s | 1280 |
| Effv2l | 1280 |
| Convnext Small | 1024 |
| Convnext Large | 1024 |


Note: The arcface can tackle about 60% of the TMAs. So even the second stage is heavy, it doesn't cause timeout.

### Training:  

1. Tile patches using the official segmentation masks. In addition to the official 5 categories, classify patches of  `healthy` and `dead` as `Other` class. Instead of training with the official TMAs , which has limited number, use them for subsequent validation and retrieval.  
2. Use the first-phase models to generate pseudo-label patches on the remaining 300+ WSIs without mask.  
3. Inherit the weights from the first or second step, training only the last layer of the backbone and the arcface head.  
   
## WSI Approach  
   
### Summary of Key Points:  
1. **Train the feature extractor using the TMA pipeline**.  
2. **Synthesize `Other`  WSI** during the training process.  
3. **Integrate different magnification scales.**  
4.  Rank the DataFrame of WSI with the number of pixels, and process the WSIs with multithread can greatly speed the the WSI processing  procedure.
   
### Inference:  

1. Ensemble of two resolutions of feature extractor: 3072 resized to 768, and 1024 without down-sampling. After extracting features, apply DTFD-MIL.  
2. For speed up the image processing procedure, I only use the center region of each 3072 tile, so that I only need to crop patch from the WSI once. 
 
| Model | Resolution | Number folds |
| --- | --- | ---|
| Convnext Small | 3072 resize 768 | 2 |
| Effv2s | 3072 resize 768 | 3 |
| Effv2s | 1024 | 4 |


### Training:  

1. Train the feature extractor using the TMA pipeline.  
2. Extract features using the feature extractor.  
3. Predict the probability of 'Other' on all patches using the feature extractor, and **create an `Other pool`** with patches that have a high probability of 'Other'.  
4. During the training of DTFD-MIL, dynamically synthesize some `WSI` from the `Other pool` each epoch.  
   - Training DTFD-MIL serves as a validation for whether the TMA pipeline models truly learned useful features. If  we use the pretrained weights from ImageNet, DTFD-MIL requires up to 200 epochs to converge. In contrast, using models trained on TMA, the MIL Head may take as little as 1 epoch and at most 20 epochs to converge.

## Something I don't have time to try but I think may work
1.  Arcface for WSI. 
2.  Large transformer pretrained on Large set of Slides. In fact I tried PLIP at the early stage of this competition,  But I didn't dig deeper.
3.  Better retrieve strategy for arcface.
4.  More resolution for WSI . I tried to add `6144 resize to 1024` into my final pipeline. But the notebook crashed.
5.  Ensemble of MIL head
6.  External data.

## Todo: 

This post is ~~not completed . Here are todo lists today~~ almost completed :
1.   ~~Model details~~
2.  ~~Citation of some methods in the post.~~
3.  ~~Discussion on MIL~~
4.  ~~Some figures about the pipeline and EDA
~~
If possible, I may add more ablation study in two weeks.

## Discussion and Citation
1.  [DTFD-MIL](https://arxiv.org/abs/2203.12081) is a strong and robust baseline for MIL, which follows ABMIL. 
2.  I think MIL methods are not sensitive to the position and the number of patches, which is observed in my experiment. That's why I randomly synthesized some `WSI` from the `Other pool` each epoch for WSI. And reduce the number on `1024 resolution wo down-sample` for submission.  This is also proved in [this paper](https://ieeexplore.ieee.org/document/10219719/authors#authors). 
3. The reason why I ensemble multiple resolution for WSI: if you check the [clarification on WHO](https://www.pathologyoutlines.com/topic/ovarytumorwhoclassif.html). You will find that Pathologists distinguish the subtypes of Ovarian cancer on different magnification. So I believe ensemble multiple resolution for classifying a WSI should be very important. 
4.  About Larger transformer-based model: I believe the competition needs model with better generalization. And larger models are often more robust. At the early stage of this competition, I takes some time to test PLIP offline, but the CV is not good. Also, since I decide to solo this time, So I have to spend my limited time on the direction I am most confident about, solving this task in a more traditional and solid way. 

## Closing Thoughts: My Journey to Grandmaster  
   
The path to becoming a Grandmaster has been lengthy, filled with challenges, and an experience I will cherish for life. I am incredibly thankful for the support from my teammates in past competitions, as well as the unwavering encouragement from my family and girlfriend throughout my journey. The final step to becoming a Grandmaster, achieving a solo gold medal, has been particularly solitary and tough. This was my fourth attempt at a solo gold medal. if I failed this time, it might have been three years, ten years, or perhaps never before I'd have the chance again, as I am about to graduate with my master's degree and start my career in a busy company. Fortunately, I have realized the dream I had three years ago and have now brought my student years to a close with the title of Grandmaster. Wishing everyone a Happy New Year!
