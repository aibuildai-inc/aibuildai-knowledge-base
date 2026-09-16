# 6th Place Solution: Detector-Free SfM & Transparent Scene Trick

Competition: image-matching-challenge-2024
Rank: #6
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/511291

# 1. Intro
We are delighted to participate in the Image Matching Challenge 2024. We would like to express our gratitude to the organizers, sponsors, and the staff of Kaggle for their efforts in making this competition possible. We also thank all the participants for their valuable suggestions and assistance.
Our team consists of Hao Yu, Xingyi He, Dongli Tan, Sida Peng, and Xiaowei Zhou. We are affiliated with the State Key Laboratory of CAD&CG at Zhejiang University. I’m very grateful for the hard work and dedication of our teammates in this competition.
# 2. Overview
Our final solution involves using [Detector-free Structure from Motion (DFSfM)](https://github.com/zju3dv/DetectorFreeSfM) for general scenes and an image order recovery strategy for transparent scenes. For general scenes, DFSfM continues the winning strategy from [our IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407), and for transparent scenes, we identified the patterns in the camera trajectories for transparent scenes in IMC 2024. By uniformly sampling the camera center positions on a circular camera trajectory and then recovering the order of each image in relation to the camera's position on the trajectory, we were able to estimate the 3D poses of the images.
# 3. Method
## 3.1 Pipeline for general scenes

Similar to IMC 2023, we utilized the DFSfM, a coarse-to-fine Structure from Motion (SfM) framework. 
The original intention behind the design of DFSfM was to address the multi-view inconsistency issues caused by the detector-free matcher LoFTR. We found that dense matchers like DKM and RoMa still have this issue and that they generate a large number of dense 2D points, significantly increasing the computational load. 
Therefore, it is necessary to merge matches for each view using a confidence-guided merging method, sacrificing some matching accuracy to improve consistency, and then refine the tracks after SfM. We used the merged matches to reconstruct a coarse SfM model. 
Subsequently, we refined the rough SfM model through a novel iterative refinement pipeline that iterates between an attention-based multi-view matching module and a geometric refinement module to enhance reconstruction accuracy. 
Due to the time constraints of the competition, we also employed a "lightweight" sparse feature detection and matching method to determine the image rotation and the final overlap areas between pairs of images, where the dense matcher (DKM, RoMa) will be executed.
### 3.1.1 Construct Image Pairs
Pairs from Retrieval (NetVLAD): retrieval involves using an image retrieval method to select k relevant images for each image. Here, we did not observe significant differences between different retrieval methods. 
### 3.1.2 Matching
#### 3.1.2.1 Roation Detection (See 2.2.1 in [our IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407))
#### 3.1.2.2 Overlap Detection (See 2.2.2 in [our IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407))
#### 3.1.2.3 Sparse Matching + Dense Matching
For scenes without highr resolution images, we used Superpoint + Superglue for sparse matching, and RoMa for dense matching instead. Due to the original RoMa being too time-consuming, we replaced the feature extraction backbone of RoMa with vit-b and retrained a model. The results showed that our retrained RoMa could achieve a speed close to DKMv3, and the local evaluation indicated a significant improvement in performance compared to DKMv3.
For scenes with high-resolution images, we found that both the original RoMa and our retrained RoMa did not perform well, so we adopted DKMv3 as the dense matching model.
### 3.1.3 Multi-view inconsistency problem for dense matching(See 2.3 in [our IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407)）



Last year, when we used the semi-dense matching model LoFTR, the coarse-to-fine matching process of LoFTR resulted in the simultaneous existence of grid-level points and pixel-level points, which caused a multi-view inconsistency problem. This year, we adopted dense matching models such as DKM and RoMa. Although the matches generated are all at the pixel-level, since DKM and RoMa sample matches from flow, they also have the multi-view inconsistency problem. Therefore, we also addressed this issue through a coarse-to-fine architecture.

### 3.1.4 Coarse SfM

#### 3.1.4.1 Confidence-guided Merge(See 2.4.1 in our [IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407)）

#### 3.1.4.2 Mapping twice

Based on the merged matches, we perform the coarse Structure from Motion (SfM) using COLMAP. We drew upon the experience from IMC 2023 (thanks to [3rd Place Solution - Significantly Reduced the Fluctuations caused by Randomness!](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417191) ) and found that after the initial COLMAP reconstruction, by relaxing the parameters of COLMAP and running it again, and then selecting the model with a greater number of registered images, this brought us  improvement.

### 3.1.5 Iterative Refinement (See 2.4.1 in our [IMC 2023 solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417407)）

### 3.1.6 Results

Here are our results for LB and local validation score.

| **Method**                                        | **Private LB** | **Public LB** | **Val (avg.)** | **pond** | **church** | **dioscuri** | **lizard** | **multi-temple** |
| ------------------------------------------------- | -------------- | ------------- | -------------- | -------- | ---------- | ------------ | ---------- | :--------------- |
| DFSfM(SuperPoint + SuperGlue + DKMv3)             | 0.164          | 0.171         | 0.399      | 0.394    | 0.195      | 0.480        | 0.495      | 0.428            |
| DFSfM(SuperPoint + SuperGlue +Our Retrained RoMa) |  **0.167**         | **0.175**         | **0.470**      | 0.487    | 0.190      | 0.542        | 0.743      | 0.390            |

Due to the large number of images in "pond" and "lizard", we sampled around 100 images for these two scenes.

## **3.2 Pipeline for transparent scenes**

We made extensive efforts and found that matching combined with COLMAP completely fails for transparent scenes. By observing the characteristics of the camera center trajectory in local transparent scenes, and according to the IMC 2024 evaluation metrics (the trajectory of the camera center, rather than specific rotation and translation), we designed a unique processing strategy for transparent scenes: generating a circular camera trajectory and recovering the order of the images.

### **3.2.1 Validate the idea on local transparent scene**

The images provided locally include the order of each image within the scene in their names, so we generated a circular trajectory and uniformly sampled the camera center coordinates according to the image order. By doing this, we found that mAA for the cylinder and cup in our local evaluation reached over 0.9.

| **scene** | **score** |
| --------- | --------- |
| cylinder  | 0.919     |
| cup       | 0.995     |

### **3.2.2 How can we distinguish these transparent scenes on Kaggle ？**

Based on the characteristics of the content in transparent scene images to segment (foreground and background). We used the [tokencut](https://github.com/YangtaoWANG95/TokenCut) segmentation model to perform foreground segmentation on each image. If the foreground area segmented from all images is roughly consistent, it indicates that the camera trajectory for this scene is approximately circular, and we mark this scene as a transparent scene.



### **3.2.3  How can we recover the image order on Kaggle?**

After distinguishing the transparent scenes, how to restore the image order on Kaggle is a challenge because the image names in the Kaggle dataset are garbled, so we can't directly sample the camera center of each image on the generated circular trajectory as we did locally with known order. Therefore, a specialized method is needed to restore the image order and place each image in its correct position within the scene.

We designed a strategy based on the *Image Similarity Matrix + TSP* algorithm to restore the image order. For instance, considering a scenario with images labeled 1, 2, and 3, which are situated on a circular path, the objective of the optimization task is to maximize the similarity among the image pairs: 1 and 2, 2 and 3, as well as 3 and 1. Thus, this problem can be formulated as a Traveling Salesman Problem (TSP), which is about finding the shortest possible route that visits a set of cities and returns to the origin. (The *Image Distance Matrix* is equal to *1 - Image Similarity Matrix* ).

After constructing the optimization problem, the key step is to estimate a similarity matrix that is closest to the ground-truth. We tried two methods.

(1)We used a retrieval method (NetVLAD) to calculate the global feature similarity of images and build the similarity matrix.

(2)We employed a Sparse Matching + RANSAC approach, using the number of matches generated between two images to represent their similarity.

We validated the recovery effect locally and also ran tests on Kaggle based on previous best approach (DFSfM with our retrained RoMa model)
Local Transparent Scenes:
| Method\Scene    | cup   | cylinder |
| --------------- | ----- | -------- |
| Random Recovery | 0.081 | 0.066    |
| NetVLAD         | **0.101** | 0.242    |
| SP + SG         | 0.020 | 0.112    |
| SIFT + NN       | 0.030 | 0.606    |
| ALIKED + NN     | 0.056 | **0.919**    |

Kaggle Submissions:
| Method                   | Private LB | Public LB |
| ------------------------ | ---------- | --------- |
| DFSfM(RoMa)              | 0.167      | 0.175     |
| DFSfM(RoMa) + NetVLAD    | 0.165      | 0.171     |
| DFSfM(RoMa) + SP + SG    | 0.189      | 0.199     |
| DFSfM(RoMa) + SIFT + NN  | **0.201**      | 0.207     |
| DFSfM(RoMa)+ ALIKED + NN | 0.191      | **0.225**     |

In the end, we chose the DFSfM (RoMa + ALIKED + NN) approach, which achieved the highest score of **0.225** on the Public LB, with a Private LB score of **0.191**. Regrettably, although the DFSfM (RoMa + SIFT + NN) approach only scored **0.207** on the public LB, it reached the highest score of **0.201** on the Private LB.

# **4. Ideas tried but not worked**

## **4.1 For general scenes**

### **4.1.1 Other image retrieval methods**

- Anyloc
- Eigenplaces
- Rotation detection before image retrieval

We attempted to use the latest retrieval methods such as Anyloc and Eigenplaces, and found that the results did not show significant differences. At the same time, we also tried to add image rotation detection before retrieval, and found that the results did not improve.

### **4.1.1 Other image matching methods**

- DeDoDe + LG
- DISK + LG
- ALIKED + LG

We also tried some other sparse matching methods, such as ALIKED + LightGlue, DISK + LightGlue, DeDoDe + LightGlue, and found that the results did not significantly improve. Both the local and public leaderboard results were slightly worse than those of SuperPoint + SuperGlue. We believe the possible reason might be that we were unable to adjust the parameters correctly.

## **4.2 For transparent scenes**

- DUSt3R



We attempted to handle the special case of transparent scenes using DUSt3R and found that, like other matching methods, it was unable to effectively process transparent scenes.

# **5. Acknowledgment:**

Once again, I would like to express my gratitude to the organizers for their contributions to this competition, and I appreciate the hard work and dedication of my teammates and all the participants.
