# [10th] Place Solution for the  Image Matching Challenge 2024 - Hexathlon

Competition: image-matching-challenge-2024
Rank: #10
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/515089

I am thrilled to share my experience participating in the Image Matching Challenge 2024 - Hexathlon. First of all, I wholeheartedly express my appreciation to the Kaggle platform, the organizers @oldufo and @eduardtrulls, and the sponsors. Secondly, I would like to thank all the kaggler. Throughout the competition, I have found the insightful posts and code in the discussion and code area to be incredibly beneficial for my learning and progress. I am inspired and would now like to share my solution with the community, hoping to offer some assistance and inspiration to other participants. 
Below is my solution for Image Matching Challenge 2024 - Hexathlon. 
## Background 
The objective of this contest is to create detailed 3D maps from collections of images captured in a variety of contexts and settings. Participants are tasked with crafting a model capable of producing precise spatial depictions, irrespective of the origin of the images—be they aerial photos from drones, shots within thick woodlands, scenes from the dark of night, or any of the six distinct problem types. 
[https://www.kaggle.com/competitions/image-matching-challenge-2024/overview](https://www.kaggle.com/competitions/image-matching-challenge-2024/overview)
[https://www.kaggle.com/competitions/image-matching-challenge-2024/data](https://www.kaggle.com/competitions/image-matching-challenge-2024/data) 
## Method 

## Overview 

My processing flow consists of five parts, Image Rotation Detection, Image Matching, Image Keypoint Extraction, Image Keypoint Matching, and Image Keypoint Fusion. I tried multiple sets of image matching hyperparameters, image matching models, and image keypoint extraction models locally. Considering the limitations of online computational resources and reasoning time, I finally chose Aliked and Affnet+hardnet for keypoint extraction, lightglue and adalam for keypoint matching. For the matching of transparent images, which is the difficult part of the competition, the number of key point matches is increased by cropping the image, thus improving the score.

## Dataset 
train dataset 
| scene | total count | size | max count |
| --- | --- | --- | --- |
| pond | 1117 | Width 576, Height 1024 | 877 |
| lizard | 711 | Width 580, Height 1024 | 284 |
| church | 110 | Width 768, Height 1024 | 92 |
| dioscuri | 70 | Width 1024, Height 768 | 27 |
| multi-temporal-temple-baalshamin | 68 | Width 1920, Height 1440 | 10 |
| transp_obj_glass_cup | 36 | Width 4608, Height 3288 | 36 |
| transp_obj_glass_cylinder | 36 | Width 6048, Height 4032 | 36 |

 <br/>
Observing the training set images based on [EDA](https://www.kaggle.com/code/moritake04/eda-imc2024-preview-all-images), it was found that there were a lot of rotated images in the dioscuri scene, my considerations were to perform rotation detection and use the results of the rotation detection for correction as well as to consider the use of a feature extractor that is not sensitive to rotation. 

 
## Image Retrieval
In order to match all pairs of images of a scene in an exhaustive way, the hyperparameters of the open source baseline are tuned. 
## Feature Extraction 
As mentioned in the dataset analysis section, one of the first things I noticed in the dataset was that certain scenes contained a large number of rotated images, and I tried to solve this problem in two ways: 
1. use rotation invariant feature matchers (AffNet/HardNet). 
2. A lightweight orientation detector is used to detect the rotation angle and rotate the image pair accordingly so that the two images have similar orientations. My consideration is that the rotation also affects the matching between the images because the rotation changes the distribution of the pixel points on the x, y axis. 
 
For keypoint extraction, the combination of ALIKED+LightGlue and AffNet was finally used because ALIKED was a very slow feature extractor in the pre-competition experiments, yet had a better score performance compared to other feature extractors (DISK,SIFT). Therefore, the pre-competition attempts were parameter tuned for ALIKED, including num_features,min_matches,resize_to etc. 
After reading the winning solutions of the 2023 competition, I learned that feature extractors such as keynet, affnet, etc. appeared in many of the winning solutions, and I implemented my own affhardnet from the 2023 open source code for feature extraction. Unfortunately, affnet, while rotationally robust, is also a slower feature extractor, which is a considerable challenge for possible model fusion. 
Thanks to [https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline](https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline) for the parallelization of the image matching and COLMAP processes, as well as Aliked's speed optimization, which made model fusion possible. 
Key feature extraction for transparent images is a important problem, I have a relatively simple treatment in this piece, for transparent images, I found that performing center cropping can improve the matching effect to some extent. 
## Feature Matching 
Use AdaLAM to match all possible pairs and merge the results with lightglue matches. AdaLAM has a number of parameters that can be tuned, such as  force_seed_mnn, search_expansion, ransac_iters, but due to the inherent stochastic nature, I don't think the gain from tuning these hyperparameters is measurable. 
## Incremental Mapper 
Based on the same reasoning, I used COLMAP's incremental mapper for the reconstruction, using almost the default parameters except that min_model_size was set to 3. 
## Ensembles 
Based on past experience, model fusion can lead to significant enhancements, especially with different models. My initial expectation was to have the Aliked+lighteglue matching results fused with Affnet using a different feature extractor, as most of the IMC2023 winners did, and strangely when I chose the three-model fusion, the commit timeout expired. In the end, I was left with the option of submitting the results of the two-model fusion. Referring to the model parameters shared by the IMC2023 contestants, simple parameter adjustments were made and used as the final submission. 
## What didn't work
When selecting different HardNet weights, there is no significant change in local validation. Using different image embeddings, there is no notable variation in local validation. 
## Sources 
- https://www.kaggle.com/code/motono0223/imc-2024-multi-models-pipeline
- https://www.kaggle.com/code/moritake04/eda-imc2024-preview-all-images
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416873
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416918
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416816
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417045
- https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417002

##Submission

##Code
I cleared the winning code (removing unnecessary comments and functions) and the link to the cleared winning code is as follows, Note that the code is inherently random, and the results will vary each time the code is run.

Cleaned Winner Code
https://www.kaggle.com/code/kirvk013/fork-of-imc-2024-multi-models-pipeline-523c69
Local Test Code
https://www.kaggle.com/code/kirvk013/fork-of-local-cv?scriptVersionId=185634850
