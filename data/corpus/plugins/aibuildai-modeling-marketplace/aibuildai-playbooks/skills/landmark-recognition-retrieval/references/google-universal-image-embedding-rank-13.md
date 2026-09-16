# 13-th place solution [Knowledge Distillation helps a lot]

Competition: google-universal-image-embedding
Rank: #13
Source: https://www.kaggle.com/c/google-universal-image-embedding/discussion/359341

Congrats to winners. Little pity just silver medal.
My solution can be summarized as following:
1. Dataset: Products10k, google landmark, deepfashion, MET Artwork Dataset, Omni, etc ... (waste lots of time in choosing dataset so that have no time to try more ideas.🙁)
2. Powerful Openclip-Vit-H backbone, linear projection layer(FC,Dropout,BN), arcface loss function
3. frozen and finetune training strategy: freeze 2/3 transform blocks of Clip and fine-tune 1/3 transform blocks(lr=5e-5) while large learning rate(lr=5e-3) to train the linear projection layer.
4. **Relational Knowledge distillation**, since the embedding dimension is 256 in the above steps. Directly using adaptive average pooling will harm the performance of the model. Thanks to the [this](https://www.kaggle.com/competitions/google-universal-image-embedding/discussion/336616). I found [this paper](https://arxiv.org/abs/1904.05068?context=cs.LG). This paper propose the relational knowledge distillation strategy and work well in several image retrieval tasks. Following this paper, I directly insert a linear projection layer after the original linear projection layer(reduce 256->64) and train the student network.

Distillation can bring 0.015~0.03 LB compared to directly adaptive average pooling.
Vit-L avgpooling: 0.608, distillation: 0.623
Vit-L-336 avgpooling: 0.628, distillation: 0.643
Openclip-Vit-H avgpooling: 0.641, distillation: 0.663
(all in publice LB)
**( BUT!!!! Maybe directly trained 64 projection layer also work well, I had no time to do more experiments...☹️). (solved)**
update:  code (naive version without collating) available at https://github.com/StarrySky-SHT/GUIE-13th-place-solution 
update: **directly train 64 projection layer scored 0.644/654 in public/private LB while distillation scored 0.663/0.666 in public/private LB, so it means knowledge distillation do improve the performance of the model.✔️**
