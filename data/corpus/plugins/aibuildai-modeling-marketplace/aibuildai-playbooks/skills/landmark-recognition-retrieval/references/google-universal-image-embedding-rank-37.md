# 37-th solution - Self Supervised Approach

Competition: google-universal-image-embedding
Rank: #37
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359402

I know that is not the best solution but I want to share this very simple approach to score 6.37 in private LB. This is the recipe:
1) Freezed backbone of openCLIP ViT-H-14 on LAION-2B

2) Trainable linear head (no batch normalization, no dropout)

3) Training in self supervised learning on 130k dataset:
https://www.kaggle.com/datasets/rhtsingh/130k-images-512x512-universal-image-embeddings

- contrastive loss:
https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss
(NTXentLoss also produce similar results)

- image batch are simply created as:
[batch, augment(batch)] (simclr style)
where augment is composed by:
randomcrop, randomrotation, randomblur, colojitter, cutout

- Total batch size of 512 and adam as optimizer

- 30 epochs

4) TTA using square padding (zero filled) and simple 224 resize

To conclude: maybe with a larger dataset the score can be improved but I have no time to test it. Hope that this could be helpful.
