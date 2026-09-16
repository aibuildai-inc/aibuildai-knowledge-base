# 15th Place Solution

Competition: image-matching-challenge-2025
Rank: #15
Source: https://www.kaggle.com/c/image-matching-challenge-2025/discussion/582810

First of all, thank you very much for organizing this wonderful competition. I sincerely appreciate both the organizers and participants.

Unfortunately, I dropped from 9th place to 15th, but here is the approach I took:

## Base solutiion

- Model: Feature extraction using Aliked, feature matching with LightGlue

- TTA with num_features = 8192 (image sizes: [1024, 2560, 1536, 2048])

- Pre-processing: Images were rotated in advance to determine the angle for each pair

- Expanded the search range by setting min_pair to 60–70 (increased both public and private scores by approximately 3 points compared to min_pair = 20)

- Match filtering: min_matches set to 20 (100 was effective in CV tasks but lowered leaderboard score)

- Detection threshold: aliked_detection_threshold = 0.001 (smaller values tend to reduce LB score variance)

## Various speed optimizations

- Multi-threaded reconstruction and other tasks (parallel CPU and GPU processing)

- Parallel execution of Aliked and LightGlue on two GPUs

- Set LightGlue's "width_confidence" to 0.9 for faster processing.

- Did not extract feature points for unused angles


My best private score was achieved with min_pair = 60.

Thanks for reading!
