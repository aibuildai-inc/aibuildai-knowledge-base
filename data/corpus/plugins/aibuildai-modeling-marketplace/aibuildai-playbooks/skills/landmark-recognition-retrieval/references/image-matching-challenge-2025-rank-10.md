# 10th Place Solution

Competition: image-matching-challenge-2025
Rank: #10
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/582898

**Update:** @tmyok1984 has shared the details of experiments for VGGT in [this thread](https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/582968), so please also check it!
(Note: VGGT is not integrated in below solution)

-----------------------------------------------------------
First of all, I’d like to express my gratitude to the host team and all the staff members for organizing such an exciting competition and providing outstanding support throughout the challenge.

This was my first time participating in the IMC, but thanks to the many insightful past IMC solutions, I managed to complete this challenge.
Also, competing with strong participants really kept me motivated. I’m truly grateful to all of you.

Our approach was strongly supported by @tmyok1984's leadership and the wide range of experiments & discussions.
And also we get a great inspiration from the IMC2024 solutions by @jooott and @sugupoko, which influenced core ideas in our pipeline.
I really appreciate their support and teamwork during this challenging competition.

In developing our solution, **I especially focused on selecting image pairs with high accucary** before feature matching.
There’s almost nothing particularly special about the pipeline after the image pair selection — it’s a straightforward process using ALIKED-LightGlue for feature matching and pycolmap for camera pose estimation.
We made a shake-up in the private leaderboard, but I think this was because our pair selection strategy fit well fortunately.

<br>
## 1. Key steps

### 1.1. Determine Top-k for Selecting Image Pairs (Dynamic Top-k)

- First, we **dynamically adjust the number of top image pairs (top-k)** for each dataset.
- In visually confusing scenes like fbk_vineyard, I noticed that incorrect image pairs are often formed, which can lead to mixing data from different clusters in the model. 
Therefore, for scenes with high visual similarity, to improve the accuracy of selected image pairs, we **set a very small top-k value (e.g. k = 3)** to strictly select only neighboring image pairs—similar to a MST(minimum spanning tree). The similarity score was calculated based on DINOv2 patch features.
- In contrast, for scenes like amy_gardens, where images were taken at different times or with different cameras, there may be multiple similar images of the same place. In those cases, we used a larger top-k to keep Recall.

**Note:**
- Regarding the above strategy, I think using a very small top-k has some risk: it can break the cluster into parts wrongly. 
For example, if there are many images from the strictly same location, they may connect only with each other, and not with nearby views.
Therefore, ideally, I think it is better to set a threshold for similarity score instead of top-k. 
- However, setting good thresholds is hard and depends on the dataset, so in this pipeline, I accepted this risk and used this top-k approach.
- I also assumed that in scenes like fbk_vineyard, it’s unlikely that many images were taken from exactly the same viewpoint. (It is useless act)
Therefore, I estimated the risk of cluster breakage to be relatively low.
    - But it is easy to intentionally include duplicate or nearly identical images....
    - In such cases, additional measures may be needed—for example, if there are image pairs with too much overlap, increase the top-k value accordingly.

<br>
### 1.2. Get Top-k Image Pairs Ranked by Keypoint Feature-based Similarity

- Second, we select image pairs based on the top-k value computed in Step 1.
Instead of using visual features like DINOv2 or MegaLoc, we rely on a **custom similarity score based on KeyNet-AdaLAM**.
    - I found this KeyNet-AdaLAM in [out teammates jooott and sugupoko's pipeline in IMC2024](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/510295), and moreover it was originated from [excellent solution from the 5th-place team in IMC2023](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/417045).
- Global descriptors could not distinguish between visually similar scenes, however, by carefully looking at small details, I found that adjacent images could be identified—like puzzle pieces that fit together.
Thus, I think keypoint-based matching was considered a more rational choice for extracting accurate image pairs.
- KeyNet-AdaLAM was selected due to below advantages:
    - Robustness to scale and rotation (via AffNet & OriNet)
    - Fast exhaustive matching for all image pairs
    - Higher precision in pair selection than other sparse matching method (e.g. ALIKED + LightGlue)
- The similarity score was heuristically designed, based on the average descriptor distance of inliers and the number of inliers.
Using this score, **we could accurately extract true neighboring pairs in fbk_vineyard**, as confirmed by the top-4 examples shown below.
(Since the dataset uses sequential image file names for adjacent views, the correctness of the selected pairs is evident)

- By generating the image pairs in such a strict way, we were able to estimate the camera poses for fbk_vineyard fairly accurately.
(As for split3, there are opposite-view pairs, and it was difficult to recognize them as part of the same cluster with this pipeline...)


**Note:**
- To improve the accuracy of image pair matching, increasing the `ransac_iters` parameter in AdaLAM is also important (we used 384).
- However, higher values may cause GPU memory errors, so be careful.

<br>
### 1.3. Extract Features by ALIKED & Match by LightGlue

- Next, we perform feature matching on the extracted image pairs using ALIKED + LightGlue (`resize_to`: 1024, `max_num_keypoints`: 10000).
- There’s not much customization in this section, but I adopt a cropping method based on DBSCAN, inspired by the [excellent solution from the 1st-place team in IMC2024](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/510084).
    - I initially assumed that this approach might have limited impact specifically on walkthrough-type scenes like fbk_vineyard, but it still showed some positive effects in both CV and LB, so I decided to include it in our pipeline. (Especially, I remember it working well on the amy_gardens scene.)
- Finally, we combine the matching results from the full images and the cropped regions, and pass them to the final step. 

<br>
### 1.4. Geometric verification & Camera pose estimation (by pycolmap)

- Finally, we remove outliers and low-confidence pairs based on MAGSAC++, save the cleaned matches to the database, and run pycolmap for camera-pose estimation.
- Because we already have well-curated image pairs, we don’t use match_exhaustive().
- Parameter tuning in pycolmap gave only small gains, but we saw some benefits by keeping `ba_local_max_num_iterations` higher.


<br>
The final metric scores for each training dataset in IMC2025 are shown below.
"stairs" was especially difficult to handle with the current pipeline...
|  | ETs | amy_gardens | fbk_vineyard | stairs |
| --- | --- | --- | --- | --- |
| score (%) | 74.98 | 40.00 | 62.65 | 5.26 |
| mAA (%) | 60.61 | 25.00 | 45.62 | 2.78 |
| clusterness (%) | 100.00 | 100.00 | 100.00 | 50.00 |


<br><br>
## 2. Other Ideas
### 2.1. Multi-processing
- I referred to @tmyok1984’s solution from last year, and applied parallel processing using two GPUs (T4 × 2).
Ideally, CPU and GPU tasks should also be processed in parallel, but in my current solution, the total runtime was acceptable without further optimization.

### 2.2 Methods That Did Not Work
- Extractors other than ALIKED (While ALIKED+DISK+SIFT showed good results in CV, they did not work at all on the LB, so I excluded them)
- Rotation Handling (Although it is generally important, it did not work well in my pipeline. I guess rotation was not a critical factor in the newly added datasets this year)
- Refiner after Camera Pose Estimation (It didn’t match well with my pipeline. However, I believe it could still be beneficial depending on the configuration)

### 2.3 Exploration of VGGT (Not imcorporate in my pipeline)
- The exploration of VGGT’s potential was mainly done by @tmyok1984 and @jooott, and we found that **[VGGT: Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt) can estimate camera poses for “stairs” scene relatively well** (still not perfect).
- Also, additional tests by @tmyok1984 showed that **VGGT also works relatively correctly for datasets with regions with sparse image coverage such as amy_gardens, and for opposite-view pairs in fbk_vineyard split3**.
- On the other hand, absolute accuracy for VGGT's camera pose seems to be not very high, so it was difficult to make it work well in LB.
- Details of these experiments has shared by tmyok in [this thread](https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/582968), so please also check it.

<br>
Thanks again to all, and I'd love to join again if the challenge is held next year!
