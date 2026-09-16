# 3rd Place Solution 😎

Competition: image-matching-challenge-2022
Rank: #3
Source: https://www.kaggle.com/c/image-matching-challenge-2022/discussion/329540

**Overview**
1. We use 4 sub-methods. SuperPoint + SuperGlue + Refinement, DISK + SuperGlue + Refinement, LoFTR, DKM.
2. Match NMS filters proximity keypoints based on matching score.
3. Adaptive augmentation matches again based on the transformed images. The transformation is computed by original matching results.


**Keypoints of our method**
1. Refinement for matching results of SG.
2. Ensemble sub-methods based on match_NMS.
3. Adaptive augmentation based on original matching results.


|Method  | Private Score |Public Score|
| --- | --- | --- |
| sp+sg+refine | 0.77974 |0.78240 |
|  (sp+sg+refine) + (disk+sg+refine) + loftr + dkm + match_nms| 0.83904|0.84268|
|(sp+sg+refine) + (disk+sg+refine) + loftr + dkm + match_nms + Adaptive_aug| 0.85866 |0.86110 |


**Team Information**
Nreal is leading the world in the revolutionary AR transformation. Widely recognized for its superior display technology and product design, Nreal has established partnerships with more than 10 world-renowned carriers and brought its product to 6 countries to date. Founded in 2017, Nreal first received worldwide recognition when it debuted Nreal Light in early 2019, the first ever AR glasses for consumers. Not only did it feature a revolutionary design, it was also the first AR glasses connected to the mobile content ecosystem. 
Welcome to join us. https://www.nreal.ai/
