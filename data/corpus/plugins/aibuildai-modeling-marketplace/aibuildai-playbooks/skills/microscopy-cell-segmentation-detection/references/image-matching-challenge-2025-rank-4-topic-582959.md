# 4th Place Solution

Competition: image-matching-challenge-2025
Rank: #4
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/582959

I’m very excited to have participated in this great competition.  
Huge thanks to the organizers and the Kaggle team for hosting such an amazing challenge. I also want to thank the Kaggle community and fellow participants for sharing valuable insights, discussions, and notebooks—it was a great opportunity to learn, experiment, and improve my skills.

As someone relatively new to this type of competition, my implementation started from [Baseline: DINOV2+ALIKED+LightGLUE](https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue) by @octaviograu, and was inspired by previous solutions from @motono0223, @tmyok1984, @igorlashkov, and their collaborators. Their work was incredibly helpful and is much appreciated.

### Overview of the solution

The core of my solution is [RDD: Robust Feature Detector and Descriptor using Deformable Transformer](https://xtcpete.github.io/rdd/) which will be presented at CVPR 2025.



#### 1. Orientation Fix

Similar to past solutions, I used [check_orientation](https://github.com/ternaus/check_orientation) to correct image orientations. However, the model produced a relatively high number of false positives. To mitigate this, I applied a confidence threshold of 0.9; only predictions with confidence above this value were accepted.

#### 2. Image Retrieval

The key challenge in this competition is to cluster a messy collection of images. This naturally leads to framing the problem as binary classification: determining whether a given image pair belongs to the same cluster.

To tackle this, I built a classifier using a linear transformer to predict whether two images are from the same scene.

Since the provided training data alone was insufficient to train a robust model, I augmented it with a much larger dataset, MegaDepth. I performed image retrieval over all combined images to generate candidate pairs, which included both true and false matches. Labels were derived based on the known scene structure.

The trained model achieved 99% accuracy on a validation set composed of both IMC and MegaDepth samples.

The model proved effective at pruning false matches. The image below shows an example of a cleaned image graph in scene ETs, after applying the model:


Initially, adding this pruning step decreased my public leaderboard score—it also removed some true pairs. To address this, I added NetVLAD-retrieved pairs to compensate. For each scene, I included:
- At least 20 pairs.
- At most 0.12 × n pairs, where n is the number of images in the scene.

This didn't improve the public score but reduce the runtime significantly and allow other tricks that can actually improve the scores to run.

#### 3. Image matching

- I used RDD to extract 8192 features using image size 1024 and 1280.
- I used LightGlue to match RDD features with filter_threshold=0.2 and width_confidence=0.99.
- Matches from 2 image sizes are merged to generate final matches.

### Things I tried that didn't work

- Used doppelgängers and its followup work doppelgängers++ to distinguish pairs.
- Combine different keypoint locations(Superpoint, ALIKED, DaD) with RDD descriptors for matching.

### Things I didn't have time to try but might work

- ROI matching that crop the original images and then rematch.
- Rotation augmentation that rotate one image in 90° increments (4 times), match each version, and keep the best result.

### Metrics

|      | Public Scores | Private Scores |
|----------|----------|----------|
| Baseline    | 46.27  | 45.16  |
| + Pair Removal | 44.89  | 45.15  |
| + Pair Removal  + netvlad (min:20; max: 0.12n)  | 46.10 | 45.48  |
| + Pair Removal  + netvlad (min:20; max: 0.12n) + TTA  | 48.02 | 47.88  |

- Baseline: orientation fix + netvlad (up to 200 pairs) + RDD (8192 feature + imsize 1024) + LightGlue
- TTA: Run RDD+LG with two different image sizes 1024 and 1280 and then merge.

### Code

[Notebook](https://www.kaggle.com/code/xtcpete/merge-rdd-lightglue/notebook?scriptVersionId=243557064)
[RDD](https://github.com/xtcpete/rdd)
[Pair Classifier](https://www.kaggle.com/datasets/xtcpete/kaggle-classifier)
[hloc with rdd and dino](https://www.kaggle.com/models/xtcpete/hloc-with-rdd/)
