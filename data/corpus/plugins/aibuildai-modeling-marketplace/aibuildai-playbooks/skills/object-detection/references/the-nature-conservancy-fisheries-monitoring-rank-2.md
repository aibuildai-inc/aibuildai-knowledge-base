# Solution Sharing and Congratulations

Competition: the-nature-conservancy-fisheries-monitoring
Rank: #2
Source: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/31538#175036

My submission with score 1.16 on the private leaderboard is a single FasterRCNN pretrained VGG16 model fine-tuned on 11-fold cross validation. The folds contained images form different cameras to generalize on new boats or cameras. Predictions from 11 models were simply averaged.

My best submission with score 1.09 was a geometric mean of above model and 10 variations in detection frameworks, CNN architectures, bbox generation, and.. random seeds. Surprisingly, those variations were tuned to public leaderboard only, no cross validation was used. Disclaimer: do not repeat this in other challenges, it's totally wrong.

Also used things that usually work:

 - NoF probability calculated as 1-sum(other_fish_probabilities).
 - Clamping probabilities to [0.02; 0.98].
 - Training/Testing images augmentation: flip, vflip, rot90.
 - Pretrained Resnet101, VGG16.
 - Bounding boxes as rectangles containing fishes / bounding boxes as squares generated from points where fish head and tails ends / bounding boxes as single rectangle merged from multiple fishes.
 - Early stopping when validation set was not used.
