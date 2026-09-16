# 1st/3rd place solution by Team smlyaka

Competition: landmark-retrieval-2019
Rank: #1
Source: https://www.kaggle.com/c/landmark-retrieval-2019/discussion/94735#latest-551277

(Updated on June 10: Priv LB was updated. Finally, our team won 1st on the retrieval challenge!)
(Updated on June 12: Our solution code and paper are available. We plan to refactor them after the CVPR conference. If you are interested in the code, I strongly recommend you to read it a few weeks later. Also, our model parameters and cleaned dataset will be available. Reproducing our work is very welcome! (code) https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution, (paper) https://arxiv.org/abs/1906.04087, (poster) https://www.dropbox.com/s/y3c3ovdiizz59j4/cvpr19_smlyaka_slides.pdf?dl=0)

----

Congratulations to all participants and winners! And special thanks to Filip Radenovic for his great paper and open-source code. We learned a lot from them.
We two worked very hard and enjoyed this competition very much.

Here is a brief overview of our solution. (Now we are working hard for preparing the detail description toward the paper submission deadline.)

1.  Automated data cleaning with using train labels, global descriptors, and spatial verification (SV). For preparing a cleaned train set, we selected only the images that it's topk neighbors in descriptor space are same landmark_id and verified by DELF+SV. This dataset cleaning is essential to train networks since the original train set is too noisy to train networks properly. To train with the cleaned dataset significantly boosts our score compared to only training with train data from previous year's competition.
2. We use FishNet-150, ResNet-101, and SEResNeXt-101 as backbones trained with cosine-based softmax losses, ArcFace and CosFace. Besides, various tricks are used such as aspect preserving of input image, cosine annealing, GeM pooling, and finetuning at full resolution on the last epoch with freezing Batch Normalization.
3. For the recognition task, accumulating top-k (k=3) similarity in descriptor space and inliers-count by spatial verification helps a lot.
4. For the retrieval task, reranking with using (3) significantly improved our score in stage2.

DBA, alpha-QE, Iterative DELF-QE (proposed by Layer6 AI team last year), Diffusion [Yang+'19] or Graph traversal (EGT, rEGT) [Chang+'19] work in stage1, but they don't work well in stage2.

Our experimental results for the retreival/recognition are attached. In table 1, all results are based on euclidean search only. No use of PCA/whitening, DBA, QE, Diffusion or Graph Traversal.

[Table1]
[Table2]
[Table3]

## Post-processing for the recognition task

Supplement to the Table 3. We treated the frequently occurring landmark as distractors and updated their confidence score in a post-processing step. Specifically, the corresponding landmark's confidence score is calculated by multiplying the frequency by -1. We treated landmarks that appeared more than 30 times as distractors.

## Things doesn’t work

- [AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations](https://arxiv.org/abs/1905.00292)
- [Combination of Multiple Global Descriptors for Image Retrieval](https://arxiv.org/abs/1903.10663)
- Label smoothing
- SPoC, MAC, R-MAC, Compact bilinear pooling
- Data Augmentation from AutoAugment paper (imagenet setting)
- Warmup learning rate scheduling
- Under Sampling
- Remove distractor by Places365 indoor/outdoor classification
- Meta classifier
- Gaussian SARE
- ... and many other methods!

Best,

Team smlyaka.

.... And I want to thank my team mate, lyakaap! Many congrats lyakaap for becoming Grandmaster ;-)
