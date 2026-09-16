# 14th place solution

Competition: image-matching-challenge-2025
Rank: #14
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583977

First, I would like to thank the organizers and Kaggle team for this exciting competition. 
My teammate @khlifimohamed and I are proud to have finished in 14th place. While this was our official ranking, we also have other submissions that weren't selected and could have placed us in the gold medal tier. Regardless of the outcome, the experience has been immensely rewarding.


**1. Overview:**



**2. Global Feature Extraction + Clustering:**

We use **DINOv2** to extract global image embeddings, which are then projected via **t-SNE** for dimensionality reduction. These low-dimensional features are clustered using **HDBSCAN**, a density-based algorithm that groups images based on visual similarity and also detects outliers. To ensure robust clustering, we defined a **confidence score 𝑠** as the harmonic mean between HDBSCAN clustering confidence and the silouhette score. Cluster assignments are accepted only if 𝑠 > 0.6.


**3. Keypoints Detection + Exhaustive Pair Matching:**

This part of our solution is largely based on the 2024 first-place solution, which we found to be highly effective and well-optimized. For local feature extraction and matching, we used **ALIKED** to detect keypoints and **LightGlue** for robust matching. As for their parameters, we leveraged two resolutions: **1280** and **1088** for the full image, and **1280** for cropped regions. The cropping is guided by DBSCAN, which identifies dense regions of matched keypoints in order to focus on relevant areas. To keep runtime practical (~6 hours), we limited the number of keypoints to 8192 per image. For geometric verification, we also adopted the custom RANSAC implementation from last year solution.


**4. Reconstruction:**

We used **PyCOLMAP v3.11.1** for incremental 3D reconstruction. 
To improve reconstruction stability and reduce randomness, we fixed the first image pair used for initialization. The **first image** is the most matched one, and the **second image** is the one with the most keypoints matched to the first. This initialization consistently increased our reconstruction quality and score. However, fixing the initial pair introduces a risk of divergence in the reconstruction process. To overcome this issue, we implemented a fallback strategy: if PyCOLMAP failed to converge, we iteratively tested alternative second images (e.g., second-best, third-best), and selected the first converging reconstruction.


**4.1. Via Clustrering:**

When the confidence score 𝑠 > 0.6 (as described in Section 2), 
we leverage the clustering method to guide reconstruction. However, clustering can be error-prone in some scenes. Therefore, we adopted the following approach:
- If s > 0.85 : we run the reconstruction on each predicted cluster.
- If 0.6 < s ≤ 0.85: we use clustering **only for pair initialization**, and we run reconstructions on all the images, followed by a postprocessing to remove the overlap between the obtained reconstructions.


**4.2. Iterative approach:**

This approach is used when the confidence score 𝑠 < 0.6 or if there was some issues when reconstructing via clustering. We run multiple rounds of reconstruction over all images, removing only those whose all paired images have already been successfully registered. The iterations continues until there are no remain images. Then, we apply a postprocessing step to merge reconstructions and eliminate overlaps.


**Our code:** https://github.com/saif-daoud/IMC-2025-14th-place-solution
**Kaggle kernel:** https://www.kaggle.com/code/saifdaoud2/pycolmap-imc-2025/notebook?scriptVersionId=243122336
