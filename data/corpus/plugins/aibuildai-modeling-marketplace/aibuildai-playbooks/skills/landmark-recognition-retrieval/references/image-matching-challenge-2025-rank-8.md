# 8th Place Solution

Competition: image-matching-challenge-2025
Rank: #8
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/582844

Thank you very much for organizing this wonderful competition. I sincerely thank the organizers and contestants. At the same time, I want to say that copilot helped me a lot, and our code relies heavily on copilot.

# Context
Business context: https://www.kaggle.com/competitions/image-matching-challenge-2025/overview
Data context: https://www.kaggle.com/competitions/image-matching-challenge-2025/data

# Our main optimization points are as follows:
1.  Use gimlightglue: PB score from 32.17=>33.75, we found that gimlightglue produces more accurate matching pairs, but fewer matching pairs.
2.  Use CLIP to replace DINO for screening: 33.75 =>36.98, CLIP performs exceptionally well in the training set, we use 0.76 cosine similarity, CLIP produces perfect segmentation in most scenes, but there is confusion between scenes in difficult scenes such as stairs.
3.  Use secondary matching, gimlightglue matches and then uses alike_lightglue (baseline) to match the matching area again: PB score increases to 41.7. This involves matching pair filtering. We first perform dbscan clustering on the matching points separately, then merge the clustering results of the two images, and finally select the points near the cluster centers of the two images for secondary matching. If clustering fails during this process, the matching pair is directly abandoned.
4.  Use loop checking to filter matching pairs that can form a loop, filter out matching pairs with a loop error greater than 30, and remove the matching pairs with the least number of matches for matching pairs with an average loop error greater than 30.
5.  Limit the number of matching pairs (take the top 1500). We found that this not only improves performance but also increases matching efficiency.
6.  Initial matching uses gimlightglue and alike_lightglue for integration, and secondary matching uses alike_lightglue.


# Useless attempts
1.  TTT: Fine-tune during testing. In order to further enhance the matching ability of Lightglue, we tried to fine-tune the model with the test data set during testing. We used the image and the transformation of the image itself for self-supervised learning (rotation of the same image, projection transformation, etc.). We tried this solution for half a month. We found that it can significantly increase the number of matching pairs in the stair scene, but too strong matching leads to more matching between scenes (so we also spent a lot of time studying the design of the classifier), and the PB score dropped after use.
2.  Classifier training: We believe that classification is the key to this image matching competition. We can use CLIP to produce perfect segmentation in most scenes, but there is some confusion in the stair scene. There are some errors that are easy to distinguish with the naked eye, but Lightglue will match the two. Therefore, we tried CNN image classification model and logistic regression model for classification respectively. We found that the classifier can work well in the training set, which can improve about two points, but PB did not improve. This solution also took a lot of time.
3. Yolo mask: We found that some building scenes in the training set contained a large number of people. We thought of using masks to remove them to reduce the interference of irrelevant scenes, but experiments found that masks would produce incorrect masks for ETs and human sculptures of some buildings. Finally, we added color checks to avoid such false masks as much as possible, but PB did not improve at all.
4.  Multi-resolution, image rotation angle correction: We tried the multi-resolution and rotation correction commonly used in previous solutions, but the PB score did not improve. It may be that there is a problem with the method we used.

# Some additional information
## 1. CLIP vs DINOv2

It is easy to see that CLIP produces a more accurate segmentation than DINO

# The location of our code
https://github.com/yangyefd/IMC2025

# Links
https://github.com/xuelunshen/gim
https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue
