# 6th Place Solution

Competition: image-matching-challenge-2025
Rank: #6
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583076

I'd like to express my gratitude to the hosts and Kaggle staffs for organizing such an exciting and practical competition!
I would also like to thank the competitors who pointed out the metric bugs, and the staff for promptly addressing them.

# Overview
* To increase the number of matching pairs with consistent orientation, the orientation of all images was detected and then aligned by rotating the images accordingly.

* As with many top competitors from last year, I primarily used ALIKED for keypoint detection and LightGlue for matching.

* Instead of using global features, I matched all image pairs and filtered the top k * log(n)/(n-1)% image pairs to extract matches within each scene (cluster) accurately. Here, k is the parameter to adjust the ratio of selection, and n is the number of images in the dataset.

* To improve mAA scores using a small number of accurate pairs, I adopted two complementary strategies: **locally**, I applied high-density crop matching by cropping regions with densely matched keypoints; **globally**, I applied image 4splits by splitting each image into four parts, which significantly increased the number of matched keypoints across the entire image.

* To mitigate mAA score fluctuations due to image scale, pair filtering and high-density crop matching were executed at multiple scales(resize_to = 1024, 1280, 1536, 2048) and ensembled into the colmap.

# Pipeline
I submitted two notebooks.

| Notebook | CV* | Public LB | Private LB|
| --- | --- | --- | --- |
| ① Best CV | 58.64 | 46.67 | 46.61 |
| ② Best LB | 54.86 | 46.98 | 45.33 |

*CV used all datasets in the train folder.

### ① Best CV



#### Steps
1. Detect the orientation of each image and rotate them to align their directions consistently.(https://github.com/ternaus/check_orientation)

2. Match for all image pairs using ALIKED (resize_to=1024) + LightGlue.

3. Select the top k * log(n)/(n-1)% pairs based on the number of matches. k is the parameter to adjust the ratio of selection, and n is the number of images in the dataset.
k was experimentally determined to achieve high scores on both CV and LB. The same rationale applies to the subsequent steps.

4. With the filtered image pairs from step 3 (k=7), match using ALIKED (resize_to=1280, 1536, 2048) + LightGlue, and again select the top k * log(n)/(n-1)% pairs.

5. From the merged results of all scales (resize_to=1024,1280,1536,2048), select the top k=1.5 pairs.

6-1. Split each image into four parts, and match using ALIKED (resize_to=1024) + LightGlue.

6-2. Crop regions with high match density from each image and match using ALIKED (resize_to=1024,1280,1536,2048) + LightGlue.

### ② Best LB (description only)

#### Steps
1. Detect the orientation of each image and rotate them to align their directions consistently.

2. Match for all image pairs using ALIKED (resize_to=1024, 1280, 1536, 2048) + LightGlue.

3. Build an undirected graph where nodes are images and edge weights are the average number of matches.

4. Cluster the graph using the Louvain method. For each cluster, select the top k * log(n)/(n-1)% image pairs (k is the parameter to adjust the ratio of selection, n is the number of images in the cluster).
The Louvain resolution parameter was set to a small value (resolution=0.1) to avoid over-segmentation.

5. From the merged results of all scales (resize_to=1024,1280,1536,2048), select the top k=1.5 pairs.

6-1. Split each image into four parts, and match using ALIKED (resize_to=1024) + LightGlue.

6-2. Crop regions with high match density from each image and match using ALIKED (resize_to=1024,1280,1536,2048) + LightGlue.

# What didn't work or were not tried
* Clustering completely based on the number of matches; scenes such as `fbk_vineyard `and `stairs` were difficult to classify even with DBSCAN or graph partitioning.
* Tuning COLMAP parameters didn’t improve the score.
* Detector-free matchers such as OmniGlue or DKM were slower than ALIKED + LightGlue and thus not adopted.

#### In the end
Last year, due to a large number of medal sellers and buyers getting banned and the resulting shifts in team population, my medal was downgraded from gold to silver.
But this year, I’m very happy to have won a gold medal!

Thank you
