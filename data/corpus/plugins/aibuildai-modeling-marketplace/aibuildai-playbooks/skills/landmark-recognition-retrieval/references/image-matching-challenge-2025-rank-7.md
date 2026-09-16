# 7th Place Solution

Competition: image-matching-challenge-2025
Rank: #7
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583184

Thanks to all the organizers and participants of the Image Matching Challenge. This competition has given us the opportunity to catch up on many 3D vision related studies and to better understand the implementation of COLMAP.

Our solution is a customized version of [the IMC'24 winning solution](https://www.kaggle.com/competitions/image-matching-challenge-2024/discussion/510084). Our new experiments, which were not part of the IMC'24 solution**,** largely ended in failure. Key features are as follows:

- Matching and rotation correction for all image pairs
- Clustering after matching

The overall pipeline is as follows:



## Clustering using similarity graph

To extract clusters from the dataset, we created a similarity graph based on inlier counts. Edges below a threshold were removed from this graph, and the resulting connected components were treated as clusters. Isolated points and small connected components were considered outliers.

## Failed Attempts

- I considered adopting MASt3R-SfM. However, when I ran and evaluated it on ETs and stairs in my local env, my experiments did not yield the expected improvements. In these experiments, I only used MASt3R's semi-dense matches and evaluated the camera poses output by MASt3R's fastNN after converting them to the competition format. While the visualized results initially appeared to estimate camera poses accurately, they didn't seem to achieve the accuracy required to surpass the evaluation metric threshold for this competition.
- VGGSfM and VGGT require a significant amount of VRAM. Therefore, I adopted an approach that only uses track refinement, as the 2nd place solution of IMC'24 employed. Due to changes in the `pycolmap` API, I had to modify a lot of code. It's possible there were bugs, but at least in my experiments, I couldn't achieve any improvements.
