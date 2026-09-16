# 4th place solution

Competition: bengaliai-cv19
Rank: #4
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/136982

We appreciate the efforts Bengali.AI and Kaggle spent on this competition, and congrats to all the winners. We are very impressed by your interesting findings, novol methods, deep understanding of the problem, the dataset, as well as the evaluation metric.

Our solution is relatively straightforward and bears similarities with some of the top solutions.

Let's call the 1295 graphemes in the train set ID (in-dictionary), and the unknown graphemes in the test set OOD (out-of-dictionary). The first step was training Arcface models and computing the feature centers of each of the 1295 graphemes. We can then tell if a test image is ID or OOD based on its smallest feature distance to every grapheme centers. In our 4th place submission, we set the classification threshold to 0.15 (cosine distance), which was estimated locally. After this ID/OOD classification, we applied the Arcface models to the ID set only and determined the component classes by the grapheme classes. For the OOD set, we trained another group of 1-head and 3-head component models and applied them to the OOD set only.

**More of the Arcface models:**
network structures: inception_resnet_v2 and seresnext101
input image preprocessing: plain resizing
input image size: 320x320
loss: Arcface
target: 1295 grapheme classification
total training epochs: 90
augmentations: cutout only

**More of the OOD models:**
network structures: inceptions, resnets, densnets, efficientnets ... ... ...
input image preprocessing: https://www.kaggle.com/iafoss/image-preprocessing-128x128
input image size: various sizes from 256x256 to 416x416
loss: CE
target: 3-head for vowel_diacritic and consonant_diacritic, 1-head for grapheme_root
total training epochs: 10 (more training harmed the OOD performance)
augmentations: mixup only
