# 4th Place Solution: ALIKED+LightGlue is all you need [Prize Eligible]

Competition: image-matching-challenge-2024
Rank: #4
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510611

I am delighted to participate in the Image Matching Challenge again, two years after my last participation in 2022. I would like to extend my gratitude to the host and the Kaggle team for organizing the competition. I also pay my respects to all the competitors who competed against each other and completed the challenge together.

This year's competition features scenes with challenging image quality compared to the previous IMC2022. It is worthwhile to evaluate the performance of the latest machine learning techniques. Additionally, I have learned that some of the distributed data includes scenes that have already been lost, such as the Temple of Baalshamin. This highlights the importance of digital archiving using 3D reconstruction technology and further underscores the social significance of this competition.

## 1. Specific Challenges Encountered

In explaining my solution, I would like to outline the key challenges encountered.

### 1.1 Rotated Images
In addition to conditions related to the shooting environment, there was a possibility that the host intentionally rotated some of the images. With the EXIF information removed, I had to rely solely on the images themselves to address this issue.

### 1.2 Transparent Scenes
The baseline code revealed that it could barely handle transparent objects. The images lacked texture and created reflections and specularities.

## 2. Solution

### 2.1 Overview

Like other teams, my solution processes transparent and non-transparent scenes separately. After processing them separately, 3D reconstruction with COLMAP is performed using the matching results obtained from each.

[overview]

### 2.2 Non-Transparent Scenes

#### 2.2.1 Keypoint Detection

Keypoints were generated and cached while rotating the images by 90 degrees at a time. ALIKED-n16 was used, and keypoints for each rotation angle were retained.

#### 2.2.2 Matching Stage
LightGlue was used for evaluating the matches. For a fixed key1, keypoints from key2 were evaluated in four patterns, and the combination with the highest number of matches was adopted. Referring to the IMC2023 2nd place solution ([link](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416873)), matches threshold was evaluated in two patterns: 100 and 125.

### 2.3 Transparent Scenes
#### 2.3.1 Foreground Segmentation - "bottle" class is all you need -

[cylinder_matching_initial]

By closely observing failure cases in transparent scenes, it was noticed that many keypoints appeared in background areas, leading to failures in camera pose estimation. To suppress keypoints in background areas, I investigated foreground extraction methods. Using the DINOv2 Segmenter ([link](https://github.com/facebookresearch/dinov2/blob/main/notebooks/semantic_segmentation.ipynb)), I discovered that the VOC2012 model assigned class5 to the foreground. According to the [link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html), class5 is assigned to "bottle". By treating this class as "transparent", I hypothesized that I could achieve high-precision segmentation.

[cylinder_segmented]

#### 2.3.2 Keypoint Detection with Original Scale
The images in transparent scenes were relatively large and all of the same size, so I decided to detect keypoints at the original scale without resizing. Considering VRAM and processing time, keypoints were detected in 1024x1024 grid units. Combining this with the DINOv2 Segmenter, keypoints were detected only in the foreground area.

[keypoints initial]

[keypoints proposed]

#### 2.3.3 Feature Matching
Given the shooting conditions this time, I determined that there was no need to search extensively for matches. Therefore, corresponding points were searched only between corresponding grids during keypoint detection. This significantly reduced the search range.

[cylinder_matching_proposed]

### 2.4 Other Tricks
#### 2.4.1 Get Pairs Exhaustive
Since the number of images per scene was not large in IMC2024, exhaustive matching for all pairs was performed instead of searching for pairs with DINOv2 or EfficientNet. This reduced the risk of missing matches due to low embedding-based similarity.

#### 2.4.2 Use *ALL* Images

> I think it's natural for the score to increase if you increase the number of images, as it makes it easier to triangulate.

Inspired by Camaro's comment ([link](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/494789#2768517)), I wondered if using images beyond those in submission.csv could facilitate easier 3D reconstruction. A simple LB probing revealed that there were images other than those listed in submission.csv in the test data's images folder. Using up to 100 image sets for validation significantly improved the validation score, so it was expected to be effective on the LB as well.

## 3. Results

After the competition ended, I conducted several late submissions to evaluate how much each additional technique contributed to improving the leaderboard (LB) score. For local validation, I used a subset of approximately 50 images generated with the following notebook.

[https://www.kaggle.com/code/tmyok1984/imc2024-validation](https://www.kaggle.com/code/tmyok1984/imc2024-validation)

| No, | Transparent trick | Exhaustive matching | Use all images | Private LB | Public LB | Val (avg.) | church | dioscuri | lizard | temple-baalshamin | pond | glass_cup | glass_cylinder |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  |  |  | 0.149 | 0.136 | 0.26  | 0.24  | 0.47  | 0.54  | 0.42  | 0.10  | 0.02  | 0.03  |
| 2 | ✓ |  |  | 0.184 | 0.171 | 0.32  | 0.24  | 0.47  | 0.54  | 0.40  | 0.09  | 0.02  | 0.47  |
| 3 | ✓ | ✓ |  | 0.186 | 0.176 | 0.34  | 0.24  | 0.52  | 0.51  | 0.42  | 0.17  | 0.02  | 0.47  |
| 4 | ✓ | ✓ | ✓ | 0.197 | 0.194 | 0.43 | 0.31 | 0.56 | 0.79 | 0.41 | 0.46 | 0.02 | 0.47 |

- Baseline (1) : Private LB=0.149 (around 300th place), Public LB=0.136 (around 300th place)
- Add Transparent trick (1+2) : Private LB=0.184 (around 8th place), Public LB=0.171 (around 50th place)
- Add Exhaustive matching (1+2+3) : Private LB=0.186 (around 7th place), Public LB=0.176 (around 30th place)
- Add ALL images (1+2+3+4) : Private LB=0.197 (4th place), Public LB=0.194 (around 7th place)

This ablation study revealed the following points:

1. Effect of Transparent trick: Adding the trick for transparent scenes significantly improved the Private LB score from 0.149 to 0.184. This demonstrated that handling transparent scenes greatly contributes to overall performance improvement.

2. Effect of Exhaustive matching: Adding the exhaustive matching method for all pairs further improved the score slightly. This indicated that the risk of missing matches due to low embedding-based similarity was effectively reduced.

3. Effect of Using All Images: Using images beyond those in submission.csv resulted in the most significant score improvement. This confirmed that utilizing additional image information greatly enhances the accuracy of 3D reconstruction.

Overall, handling transparent scenes and using all available images were shown to be key factors for ranking high on the leaderboard.

## 4. Implementation

This section shares techniques related to implementation.

### 4.1 Multiple Process / Multiple GPUs
As highlighted in previous solutions, IMC is characterized by the substantial computational cost of both CPU and GPU processing. Running the CPU and GPU in parallel can potentially double the processing speed. Additionally, using GPUs with two T4 cards allows for parallelizing GPU processing, thereby doubling the processing capacity.

Although the importance of these aspects has been mentioned in past solutions, it is very rare for reference code to be made public. As part of my contribution to the community, I have shared the code I used this time. The base part of the implementation can be used beyond IMC, so please refer to it.

[https://www.kaggle.com/code/tmyok1984/imc2024-exp556](https://www.kaggle.com/code/tmyok1984/imc2024-exp556)

### 4.2 Utility Script

When using packages not available in the Kaggle environment, offline installation is necessary. However, performing offline package installation in the submission notebook wastes submission time. To solve this problem, I always use the utility script feature. By adding a utility script notebook with pre-installed packages, necessary packages can be installed beforehand, making the submission notebook more efficient. The utility script I used this time is below.

[https://www.kaggle.com/code/tmyok1984/imc2024-install-once](https://www.kaggle.com/code/tmyok1984/imc2024-install-once)

The following link provides detailed information on how to create utility scripts, so please refer to it.

[https://www.kaggle.com/code/kononenko/pip-install-once](https://www.kaggle.com/code/kononenko/pip-install-once)

## 5. What Did Not Work (For Me)
- Symmetries-and-repeats: I spent most of the competition period on symmetries-and-repeats. Ultimately, I implemented a combination of a  [Doppelgangers classifier](https://github.com/RuojinCai/doppelgangers) and 3D geometrical verification. Although I adopted it in the final submission and it worked well locally in some cases, the LB consistently worsened during the ablation study with the late submission. I experienced the Concorde effect firsthand.
- Overlap region：In our 10th place solution for IMC2022, we used DKM-ROI ([link1](https://www.kaggle.com/competitions/image-matching-challenge-2022/discussion/328903), [link2](https://www.kaggle.com/code/tmyok1984/imc2022-ensemble-with-dkm-roi)). This time, there was an excellent publicly available notebook ([link](https://www.kaggle.com/code/nartaa/imc24-overlap-detection)), so I implemented it as well but did not achieve good results. While high matching accuracy was required in IMC2022, in IMC2024, high robustness was required, so its effectiveness was limited. Specifically, if the initial value for finding overlap was incorrect, the subsequent steps would break down significantly. In contrast, the 1st place solution  ([link](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/510084)) successfully estimated the ROI using multiple viewpoints robustly and achieved excellent results.
- Non-Maximum Suppression：This also strongly boosted our LB in IMC2022. However, ALIKED was strong enough this time that even when LoFTR, DeDoDe, and RoMa were ensembled, no improvement was observed in my local validation.
- COLMAP hyperparameters：Using the help(pycolmap) command, many configurable parameters could be seen. Various parameters were changed, but no significant effect was obtained.
