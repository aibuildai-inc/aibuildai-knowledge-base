# 5th Place Solution: Customized Scene Matching

Competition: image-matching-challenge-2024
Rank: #5
Source: https://www.kaggle.com/c/image-matching-challenge-2024/discussion/510603

First of all, thanks to the hosts for organizing the Image Matching Challenge for the third time on Kaggle. Also, congratulate the winners and wish the best to everyone who has gone through this tough journey safe and sound.

I have to admit that I attended the competition quite late. Therefore, I just did whatever I “felt” right and improvised a lot. So, you might find my write-up a little bit tricky at some points :)). Let's go!

# 1. Overview 
My solution architecture can be summarized in the following figure:
[arch]

I will split my summary into 4 main sections:
- Build local evaluation datasets
- Build a general SfM pipeline
- Customize the pipeline for each specific category
- Results

# 2. Local evaluation datasets
The original dataset is obviously too large and unrealistic to evaluate. To tackle this, I built 3 versions of subsets (by applying some random sampling strategies). The results on those 3 versions were highly correlated. Hence, I chose one as my main local validation dataset:

| Dataset/Scene |Number of samples  |
| :--- | :---: |
| Pond | 100 |
|Lizard|90|
|Church|80|
|Dioscuri|70|
|Multi-temporal-temple-baalshamin|68|
|Transp_obj_glass_cup|36|
|Transp_obj_glass_cylinder|36|


I will report my local CV results on this dataset.

# 3. SfM pipeline
For a better explanation, I will divide the pipeline into 3 modules:
- Proposing pair candidates by global descriptors
- Matching pairs in the candidate list
- Reconstruction with Colmap

## 3.1. Finding pair candidates
I used 3 pretrained models from [timm](https://github.com/huggingface/pytorch-image-models) to extract global features:
```
EVA-CLIP Base   \
ConvNeXt Base    | -->[concat]--> [fc 2560]
Dinov2 ViT Base /
```
I also customized similarity thresholds for different types of scenes (note that I used cosine similarity instead of distance). For example, with highly diverse scenes like Lizard, I used a small threshold of 0.6. In contrast, for object scenes like Cylinder and Cup, a high threshold of 0.95 would make much more sense.

## 3.2. Image matching
Following the spirit of my last year’s solution, I only focused on **detector-based** methods, because they are much more lightweight than semi-dense or dense models. Due to the lack of experiments on both CV and LB, I cannot give detailed numbers for each method, but only tell the sense of their performance in general. Here are some combinations that I tried during the competitions:

| Method                           |CV performance|LB performance|Remark                                              |
| :------------------------|:----------------:|:---------------:|:-----------------------------------|
|SuperPoint + SuperGlue|       good             |     not good      |     slow compared to LightGlue      |
|GlueStick                         |         normal        |            -             |                       slow                         |
|**SuperPoint + LightGlue** |    **very good**       |     not good      | no ideas why it is not good on LB  |
|**ALIKED + LightGlue**|good|**very good**|best on LB|
|DISK + LightGlue| good|normal||
|OmniGlue|not good|-|slow|


In addition, I also tried SIFT + NN as a light ensemble model added to LightGlue. It is claimed in [IMC23 7th solution](https://www.kaggle.com/competitions/image-matching-challenge-2023/discussion/427143) that ensembling SIFT with LightGlue could boost the performance. I think it is the case in some scenes in CV, but it harmed the LB results a lot (-0.03).

In summary, I could say that **LightGlue** is one of the best models in Image Matching problem at present (hats off to the authors)!

## 3.3. Reconstruction
This year I spent some time trying to make use of colmap better than the host baseline. Here are some of my trials:
- **Single camera for specific scenes**: good on transparent object scenes in local CV, but worsens the result on LB.
- **Manual initial pair**: I chose the pair with most matches and most keypoints and set it as the initial pair for colmap. But it made the results worse in my CV (maybe my code is wrong?!!).
- **Run Incremental Mapping multiple times**: it made the results on my CV more consistent and a little better. However, I didn’t see it had any effect on LB.

Sadly to say that I could not make any noticeable improvement with colmap.

# 4. Deal with specific categories
## 4.1. Transparent objects
I think this category decides the **gold medal**. Because when I cracked the problem, I moved from top 300 to top 10 immediately!!!
First, I will present the process of how I created this solution, hope that it will be more vivid :).
If you open one image and move to the next image-by-image, you can easily see that it is like a sequence of frames cut from a video captured by only one camera.
What a human might do to match those images? If it were me, I would focus on the **object** only and ignore the whole background. In addition, I only need to match a **sequence** (or a **ring**) of frames: (0, 1), (1, 2),...,(35, 0), which means there is no need to match (1, 20) or (3, 29)!

[ring]

By adding these two lines of code before matching the cylinder, I went from ~0.0 to **0.86** on CV with LightGlue:
```
image = image[950:2850, :, :]
index_pairs = [(i, i+1) for i in range(len(scene)-1)] + [(len(scene)-1, 0)]
```
So the problem now can be broken down into 2 smaller problems:
- Detect the object
- Find consecutive pairs
### 4.1.1. Object detection
**Method 1** (not worked): Use object detection models: I tried **Yolo** family. Nonetheless, it is hard to make the prompt for the model to find out the exact object location.
- **Approach 1**: Use **Yolov8** to produce all bounding boxes => it found no boxes!
- **Approach 2**: Use **YoloWorld** with keyword “transparent object” => not detect anything.
- **Approach 3**: Use **YoloWorld** with keyword “cup” => it could detect the cup. Yet, it is not generalized enough to apply to LB dataset.

**Method 2** (final solution): Use segmentation models:
- **Step 1**: Use **[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)** to detect all the masks in the images.
- **Step 2**: Use keypoint extractors above (SuperPoint or ALIKED) to extract object keypoints. It will produce some noise in the background.
- **Step 3**: Find the best mask: the smallest mask which contains most of the keypoints.
- **Step 4**: Find the smallest bounding box which includes the best mask.

I tried DBSCAN too. However, due to the limitation of development time, I did not go to the end to see if it could replace SAM.

[sam_mask]

### 4.1.2. Finding the best pairs
**Problem definition**:
- Version 1: Find the perfect sequence of images.
- Version 2 (simplified): For each image, find out two other best images for matching.

**Method 1** (not good): Stem from the fact that two consecutive frames would have the smallest movement among pixels, I calculated Optical Flow with [RAFT](https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html) for every possible pairs, then got the average magnitudes. I further combined with cosine similarity of Global embeddings above to build an NxN matrix:
```
pair_score[i, j] = -of_mag[i, j] + cos_sim[i, j]
```

TSP solver could not produce the perfect ring (~90% accuracy). Thus, I think it is not good enough.

**Method 2** (simpler but better): I performed exhaustive matching on all pairs. Then, build an NxN matrix with elements that are the number of matches between each pair:
```
pair_score[i, j] = n_matches[i, j]
```

Subsequently, I just picked out the top-2 largest values on each row. This simple approach astonishingly produced nearly 100% accuracy (I tried different matchers, and sometimes there were 1-2 wrong pairs).

## 4.2. Day-night
**Method 1**: Find dark images and enhance them.
Dark detection is very easy, just find those images that have the mean value < threshold (can be 65, 70).
I used **CLAHE** to enhance the dark images. Unfortunately, it downgraded my LB score by 0.01.
**Method 2**: Use [DarkFeat](https://github.com/THU-LYJ-Lab/DarkFeat) matcher.
My idea is to ensemble LightGlue with a more robust matcher for dark images. I found out that DarkFeat might be a promising candidate. Unluckily, I didn’t see any benefits of using it.

## 4.3. Symmetries-and-repeats
I assume that these scenes may have plenty of false positive matches because of their natural properties. 
**Method 1**: First, I tried tuning parameters with much more strict values (e.g., increasing the matching threshold to 0.5, etc.). Nonetheless, it didn’t show any improvement.

**Method 2**: Next, I move to a more promising method: [Doppelgangers: Learning to Disambiguate Images of Similar Structures](https://github.com/RuojinCai/Doppelgangers). The main idea of this method is:
- First, run SfM one time to get all the matching pairs.
- Use the Doppelganger model to filter out pairs that have high probabilities as false positives (doppelgangers).
It didn’t show any improvement on the church scene on my local CV.

So basically, I could not make any improvement on this category.

## 4.4. Others
The remaining categories are nature, air-to-ground, historical_preservation, temporal.
Actually, it is hard to customize these scenes, since they seem too general.
Therefore, I just applied the [rotation checking method](https://github.com/ternaus/check_orientation) to correct the image orientations. It made a huge boost on the Dioscuri scene in local CV, and added about +0.005 to the LB.

# 5. Results
**Cross validation**

|Scene|Method|Result|
|---|---|---|
|Pond|ALIKED+LightGlue|0.46|
|Lizard|SP+LightGlue|0.73|
|Church|ALIKED+LightGlue|0.18|
|Dioscuri|ALIKED+LightGlue|0.49|
|Multi-temporal-temple-baalshamin|ALIKED+LightGlue|0.48|
|Transp_obj_glass_cup|DISK+LightGlue|0.32|
|Transp_obj_glass_cylinder|ALIKED+LightGlue|0.81|

**Leaderboard**

|Method|Public LB|Private LB|
|---|---|---|
|My IMC23: SP+SG|0.104|0.111|
|SP+LG|0.099|0.115|
|SP+LG+rot|0.103|0.119|
|ALIKED+LG+rot|0.135|0.147|
|ALIKED+LG+rot+transparent_custom|0.18|0.19|
|**ALIKED+LG+rot+transparent_custom+tuning**|**0.186**|**0.195**|
|ALIKED+LG+rot+transparent_custom+tuning+dark_enhance|0.179|0.183|

# 6. Conclusion
A fun fact: I participated in all three Image Matching Challenges on Kaggle and earned medals of all three different colors 😂:
- 2022: bronze🥉
- 2023: silver 🥈
- 2024: gold 🥇

I feel exhausted after going solo in a tough competition like this. But finally, the sweet fruits have come after tireless efforts 😁.

Thank you for your reading. I hope you can find something interesting in my write-up.

Happy Kaggling!
