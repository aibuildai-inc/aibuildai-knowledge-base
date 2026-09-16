# 3rd Place Solution

Competition: image-matching-challenge-2025
Rank: #3
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583401

First and foremost, I would like to express my sincere gratitude to the organizers for hosting the Image Matching Challenge again this year. I am always inspired by the new themes you introduce each year, and I truly enjoy the opportunity to learn and experiment with new techniques.

## Overview
My solution consists of a pre-matching stage with rotation augmentation, followed by multiple matching rounds using tiled images. The core ideas of rotation augmentation and image tiling were also used in [my approach at IMC2023](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/416918).

The overall pipeline is summarized in this diagram:
.png?generation=1749222583567967&alt=media)

Like many other teams, I utilized a T4x2 GPU setup for parallel processing.

## Solution Details
### 2.1. Pre-matching
I leveraged the baseline pipeline, using DINOv2 to generate a list of image pairs with high similarity. I found the optimal parameters to be a minimum of 50-60 pairs per image (min_pair) and a similarity threshold (sim_th) of 0.2.

For each pair, I performed matching using ALIKED + LightGlue (with longside set to 960px and max_keypoints to 4096). Pairs with more than 40 inlier matches were considered valid. Since some datasets contained rotated images, I implemented a retry mechanism with rotation augmentation for pairs that initially failed to match.

### 2.2. Matching with Tiled Images
For the valid pairs from the pre-matching stage, I uniformly tiled each image into four quadrants. I then performed matching on these tiles, again using ALIKED + LightGlue (max_keypoints=4096). The optimal longside setting for the tiles was either 1024 or 1216.

A key part of my strategy was to create a set of five images for matching: the four tiles plus the original image downscaled by half. This resulted in 25 matching combinations per original image pair (5x5), which allows for more robust matching between images with significant differences in scale and perspective. While this may seem computationally expensive, the low computational cost of LightGlue made it feasible within the given time constraints. 
*UPDATE*: I've posted the implemention of the pipeline in the *comments* section below.

A reperesentative scores by the training data is summarized here. I couldn't find an effective method for the stairs dataset in the end...
|heritage	|theather_church	|dioscuri_baalshamin	|lizard_pond	|piazzasanmarco	|amy_gardens	|fbk_vineyard	|ETs	|stairs	|Average	|Public	|Private|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|88.13	|61.39	|91.73	|76.16	|60.25	|23.91	|37.88	|66.67	|4.3	|56.71	|48.97	|49.05|


Actually, I have not fully verified why this specific tiling strategy was so effective. However, I believe one reason is that it helps mitigate the issue of keypoints being concentrated in only one part of the image.

To improve time efficiency, I cached the results of the ALIKED feature extraction for reuse. After matching, I restored the keypoint coordinates to their original image space, used RANSAC to filter outliers, and then combined these keypoints with those from the pre-matching stage. The RANSAC parameters were taken from [the 2024 1st place team's solution,](https://www.kaggle.com/code/vostankovich/imc2024-1st-place-solution) and I would like to extend my thanks to them.

### 2.3. Other Points
Registering pycolmap results: My pycolmap processing was based on the official baseline. However, I paid an attention to the order in which reconstruction results from incremental_mapping are registered. My understanding for this competition is that a larger number of registered images in the largest cluster of a scene generally leads to a better score (assuming the estimated camera poses are correct). Since images can be part of multiple clusters, it's crucial to prevent the results from larger, more robust clusters from being overwritten by those from smaller clusters.
To achieve this, I sorted the reconstructed maps by the number of registered images in ascending order before registering the final poses. This modification seemed to yield a small boost on the leaderboard.

Like this:
``` 
map_info_list = []
for map_idx, cur_map in maps.items():
    num_registered_images = cur_map.num_registered_images()
    map_info_list.append({
        "original_map_index": map_idx,
        "map_object": cur_map,
        "num_reg_images": num_registered_images
    })
# Sort by the number of registered images in ascending order
sorted_map_info_list = sorted(map_info_list, key=lambda x: x["num_reg_images"])
for map_info in sorted_map_info_list:
    # (Register poses...)
``` 

Parameter Tuning: The parameters for ALIKED and LightGlue, as well as the input image sizes, had a significant impact on accuracy. Careful tuning was essential.

## What Didn't Work in My Case
- Using SIFT instead of ALIKED: While SIFT was faster, it resulted in a lower score on most datasets compared to ALIKED. I didn't have a chance to try an ensemble of ALIKED and SIFT.
- Scene Pre-Clustering: I attempted to pre-cluster images within datasets that contained multiple scenes.
I tried clustering based on a distance matrix derived from the number of keypoint matches from the pre-matching stage, but I couldn't find a reliable method to accurately segment all scenes.
Even when scenes were correctly segmented, it did not lead to a significant score improvement (for this competition's metric) unless the subsequent matching step could find enough keypoints for a robust reconstruction.
- While pairwise matching-based clustering had low accuracy, I found VGGT to be powerful from a clustering perspective, as it seems to find context among a group of dozens of images. I experimented with using SIFT feature points as query_points and identifying pairs based on the number of points exceeding vis_score and conf_score thresholds for a target image. This method achieved a higher true positive rate for pair identification than the pairwise approach in section 2.1 and successfully clustered many datasets.
However, in terms of the resulting pose accuracy and final score, it did not show a significant advantage in my setup, possibly due to the resizing.
The use of VGGT has been deeply explored and discussed by @tmyok1984 san in [this discussion post.](https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/582968) I'm learning a great deal from reading the post.
