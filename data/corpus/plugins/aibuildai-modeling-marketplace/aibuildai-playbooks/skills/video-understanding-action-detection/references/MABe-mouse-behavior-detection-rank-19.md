# 19th place - 4 Body Parts Invariant Features with WaveNet + GRU

Competition: MABe-mouse-behavior-detection
Rank: #19
Source: https://www.kaggle.com/c/MABe-mouse-behavior-detection/writeups/19th-place-4-body-parts-invariant-features-with

Since the beginning we felt like the way to solve this problem is using deep learning models even though there were barely any deep learning solutions or discussions in the forum.

Our solution consists of a K-Fold ensemble of a deep learning model per lab, where the model is a combination of WaveNet and GRU.

### For each lab and fold we trained:

- One model for single actions and one model for actions of pairs (using a window size of 64 frames). The model consists of x3 WaveNet blocks --> x3 GRU Blocks --> MLP (code below)
- We kept only 4 body parts - left ear, right ear, nose (called head in some of the labs) and tail.
- The models were trained using invariant features to prevent overfitting on coordinates data.
- We used a simple gaussian noise augmentation, up sampling on minority classes, and hard negative mining.

### Post Processing
We combined neighboring actions of the same mice/pair

### Inference
Predict using sliding windows approach with stride=2

### Ensemble
The final solution was a mean ensemble of K-Fold with K=5 and K=3 (8 models in total).


### What didn't work
- More body parts
- Per action modeling
- More features (orientation, location relative to center, etc...)


### Notes
We joined a bit late to the competition, and wanted to try out a "general" model that is not trained per lab but we didnt have enough time to try it. In hindsight it was a mistake


Github link to code - https://github.com/amita1996/MABe-Kaggle-Challenge

Hopefully the writeup is clear, feel free to ask questions
