# 2nd Place Solution: MST-Aided SfM & Transparent Scene Solution [Prize Eligible]

Competition: image-matching-challenge-2024
Rank: #2
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510499

We would like to express our gratitude to the Kaggle community and the organizers from Czech Technical University in Prague for their contributions to this competition. We also appreciate enthusiastic discussions from all participants. Congratulations to all the participating teams!
The work described here is actually a joint effort by @sunnyykk, @gdchenhao, @mayunchaoamap and @wangshengyi96. We especially thank our mentor Zhang Tao for his strong support and guidance.
I'm very glad to have been part of such an excellent team participating in IMC-2024. Throughout the competition, we have gained a lot and learned a lot.
It is an honor to share our solution with you now.

## 1. Overview
Undoubtedly, the biggest difference between this year's competition and previous competitions is the appearence of transparent and reflective scenes. After many trials, we found it challenging to develop a general solution that could handle both transparent and conventional scenes simultaneously. Therefore, we designed different methods to tackle the two types of scenes.
**Conventional Scenes**: We designed an iterative optimization SfM scheme based on the Minimum Spanning Tree (MST). We use the coarse model reconstructed from the most concise data association as a skeleton and iteratively add redundant associations to optimize the accuracy of the coarse model.
**Transparent and Reflective Scenes**: We assume that the camera captures a transparent object in a circumferential manner and focus on calculating the shooting sequence of the images. Then we place the images in the corresponding positions and orientations. Notably, we designed a new global descriptor that can effectively capture detailed information in these scenes, enabling accurate camera pose estimation.
Moreover, our code is based on the open-source solution of the 7th team [RMD-3DV](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/427143) from IMC-2023. This robust baseline, which combines technologies like feature ensemble, pixsfm, and reloc with colmap, allowed us to build upon their framework without starting from scratch. We appreciate their generous contribution.

## 2. Method
This is the pipeline of our solution:

### 2.1 Preprocessing
We start by performing rotation detection on the images and determine whether the scene is transparent or not.
- **Rotation Detection**: We use a rotation detection model to predict and correct the image rotation. However, recognizing that the model's predictions are not always accurate, we retain the original rotations if less than 10% of the images are predicted rotated.
- **Shared Camera Intrinsics**: If all the image dimensions are identical, we set all cameras to share the same internal parameters, occasionally bringing a 0.01 improvement in results.
- **Transparency Detection**: We calculate the average difference between different images to determine whether the scene is transparent or not for separate handling.

   
### 2.2 Global Features




We designed a stronger global feature descriptor that yields more reliable image pairs during the image retrieval phase.
Specifically, we developed a global descriptor combining point and patch features. The basic approach involves extracting point features (ALIKED) and patch features (DINO) from an image, then establishing a one-to-one correspondence based on their spatial relationships. Using clustering and the VLAD algorithm, we generate global descriptors. This approach allows the clustering algorithm to achieve unsupervised learning of scene features, and incorporating DINO further elevates the learning potential.
Our method outperforms NetVLAD, AnyLoc, DINO (GAP or GMP), and SALAD on VPR-related datasets. While the image retrieval in the SfM pipeline typically results in many candidate matches (30+), which provides robustness to high recall rates, distinguishing retrieval capabilities becomes less pronounced. Compared to other configurations and excluding transparent scenes, the scores of NetVLAD and our global features are as follows:

| Global Descriptor        | Private     | Public  |
| ------------- |:-------------:| -----:|
| NetVLAD  | 0.241 | 0.230 |
| **Ours**      | **0.245**    |  **0.247** |

### 2.3 Local Features
We utilized three types of local features and use the ensemble of their matching results:
- **Dedode v2 + Dual Softmax**
- **DISK + LightGlue**
- **SIFT + Nearest Neighbor**

The v2 version of the Dedode detector produces richer and more evenly distributed feature points. We selected the pre-trained G-upright as the descriptor and dual softmax as the matcher.

| Local Feature Configuration        | Private     | Public  |
| ------------- |:-------------:| -----:|
| ALIKED + DISK + SIFT  | **0.184** | 0.169 |
| SuperPoint + DISK + SIFT | 0.178    |  0.172 |
| Dedode v1 + DISK + SIFT | 0.179   |  0.177 |
| **Dedode v2 + DISK + SIFT** | **0.184**   | **0.185** |

### 2.4 MST-Aided Coarse-to-Fine SfM Solution
In the data association phase of SfM, extensive feature matching is typically undertaken to enhance the robustness and accuracy of SfM. However, increasing data associations in scenes with repetitive textures can result in more incorrect matches. 

This issue was evident in the reconstruction result of church scene in the train set.
 
To address this, we proposed a coarse-to-fine SfM solution based on the Minimum Spanning Tree (MST)
- **Stage 1**: We construct a similarity graph where vertices represent images, and edges represent similarity. By computing the MST, we obtain a globally optimal data association linking all image nodes, which is used for the first SfM. Experiments showed this method removes a large amount of incorrect associations, significantly improving coarse-grained accuracy but somewhat losing fine-grained accuracy.

- **Stage 2**: We use full data associations and the coarse model from Stage 1 providing initial camera pose priors for geometric verification. This filters out incorrect feature matches in the full data association, leading to the final model. The data redundancy maintains the coarse-grained advantages of Stage 1 while compensating for its fine-grained accuracy losses.
| SfM Solution        | Private     | Public  |
| ------------- |:-------------:| -----:|
| Direct SfM  | 0.240 | 0.253 |
| MST-Aided SfM (Stage 1 Only)      | 0.216    | 0.218 |
| **MST-Aided SfM (Stage 1 + 2)**   | **0.258**  | **0.268** |

### 2.5 Post-Processing
Following the experiences from previous competitions, we employed pixsfm to optimize the SfM model. Additionally, we deployed an HLoc-based relocalization module to process unregistered images, which typically resulted in a 0~0.01 score improvement.

### 2.6 Transparent Scenes
For transparent scenes, we tried to use various local features, including ALIKED, DISK, LoFTR, and DKMV3, but unfortunately none delivered satisfactory results. By observing the training set, we assumed circumferential camera capture of transparent targets. Using similarity graphs from global features, we calculate the min-cost path to determine the image shooting sequence, arranging all cameras in a closed loop to calculate their positions and orientations.
Interestingly, visualizing the global features of a cylinder also revealed a highly ordered 2D plan layout, aiding significantly in recovering the image capture sequence. 

The solution for transparent scenes improved scores by approximately **0.06** eventually.
| Solution       | Private     | Public  |
| ------------- |:-------------:| -----:|
| Without Transparent Solution  | 0.184 | 0.185 |
| **With Transparent Solution**   | **0.242**  | **0.249** |

## 3. Other Ideas
### 3.1 Methods that did not work
- **Enhanced First Phase of MST-Aided SfM**: By adding more redundant edges (e.g., top-k, alternative key paths) to MST, aiming to balance recall and precision in the first reconstruction phase. While private score reached a high of **0.263**, the public score was only **0.249**, so this was not selected.
- **Lightglue Matcher for Dedode**: We tested Kornia's lightglue matcher for dedode (b/g), but it performed worse than dual softmax.
- **Day-Night Challenge**: To tackle day-night challenges in pond and lizard datasets, we attempted brightness-based day/night classification, normalized feature distribution, but encountered unknown "Throw Exception" obstacles.
- **Dense Matchers**: We tested dense matchers like LoFTR, EfficientLoFTR, and DKMV3 in conventional scenes but saw no score improvement, likely due to our unoptimized parameters (matching threshold, grid size, etc.).
- **Neural Network-Based SfM**: We explored neural network-based SfM methods like Dust3R for transparent scenes, but results were not pretty well.
- **Dense Optical Flow**: Using dense optical flow to restore the image capture sequence in transparent scenes also worked but was less effective than our proposed global feature.

### 3.2 Methods that we haven't tried
- **TTA**: We didn't try TTA methods due to runtime considerations.
- **Multi-Scale Matching**: We didn't detect covisible regions or perform multi-scale local feature matching.
﻿
