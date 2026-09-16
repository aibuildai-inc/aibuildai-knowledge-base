# [21st place] - My solution to 2 silver medals in GL 2020

Competition: landmark-recognition-2020
Rank: #21
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187864

First of all, congratulations to everyone for having interesting journeys across such a challenging competition, especially to the top teams and all who gained a lot from the competition. Also, thanks to google for organizing the 3rd Landmark competitions.

For me, throughout two competitions of Google Landmark 2020, I really learnt such an amount of knowledge, and also came up with many ideas to compete in those harsh competitions. Here are things I have done.

# **Retrieval competition**
Actually, I didn’t think this competition was for me at first, because I’m kind of a newbie in tensorflow :)). However, after a bad performance in Global Wheat Detection challenge, I decided to join seriously in the competition within 13 days left.

## Architecture
First, I tried to train delf model from [tf delf](https://github.com/tensorflow/models/tree/master/research/delf), the result is no where near the host baseline kernel (only 0.12-0.13 compared to 0.271 of host baseline) so I quickly quitted this way.

Luckily, [onnx](https://www.kaggle.com/chandanverma/convert-pytorch-model-to-tf-2-2-submission-format) kernel was published and it was a big chance for a pytorch user like me that I could develop my own model. From top-down view, my final model is ensemble of 4 models, which can be represented as followed:
```
net_1 -> feature_1 |
...                |--[concat]--> [fc 4096]--> final_feature
net_4 -> feature_4 |
```
With net_i:
```
CNN backbone --> GeM pooling --> fc --> batchnorm --> arcface/cosface
```
More detailed:
```
net_1: efficientnet-b2, fc 1024
net_2: efficientnet-b3, fc 512
net_3: efficientnet-b3, fc 1024
net_4: efficientnet-b4, fc 512
```
From my results, training one more fc layer on top of ensemble net gave a 0.02 boost on final score compared to simple concatenating.
I used input images of size 256x256, which might really affect my score (compared to top solutions that use images of large size), and training only with GLDv2 clean. Also while I always use RAdam optimizer, I realize that almost all top teams use SGD, maybe I should switch to SGD in the future?
##Result
```
Best single-model score: 0.270 - 0.233 (efficientnet-b3, fc 512)
Best ensemble-model score: 0.293 - 0.260
```
I had learnt a lot from public solution of top teams after the end of the competition, and figured out somethings I could improve in the retrieval task:
- larger image size
- dealing with imbalanced dataset using class weights
- training on extended dataset: GLDv1, GLDv2 full.

Although I struggled with tf model submission type at first, it was at last very lucky for me because I didn't have to care about the post-processing phase such as local features, rescoring and reranking, ransac, etc. 
#**Recognition competition**
Maybe I was lucky in the retrieval competition, finally I had to deal with everything when coming to the recognition challenge :).
## Global feature extraction
I used entirely different set of backbones in comparison with those in retrieval task:
```
net_1: resnest200, fc 512
net_2: resnest200, fc 1024
net_3: resnest269, fc 512
net_4: resnest269, fc 1024
net_5: resnet152, fc 512
```
From my results, resnest and resnet152 gave much better performances than efficientnet (even b7). I trained with 224x224 images of GLDv1 + GLDv2 clean and fine-tuned with 448x448 of GLDv2 clean, using arcface.
## Local feature extraction
I trained delg, delf with [tf delf](https://github.com/tensorflow/models/tree/master/research/delf) again, and further trained a PCA to diminish dimension of descriptors from 1024 -> 128. However, I couldn’t beat the delg model in the host baseline kernel. My best self-trained delg is worse 0.01 than baseline delg.
So for my final score, I chose the delg baseline model to extract local features.
## Rescoring and reranking
I defined score of reranking process as followed:
```
score = global_score + local_score + recognition_score
```
Definition of my own recognition score:
```
recognition_score = max_value * (1-k/topk) ** alpha
- recognition score is based on the arcface head of my model
- max_value: max value of recognition score
- k (< topk): rank of predicted label in sorted arcface head
- topk: topk prediction of arcface, recognition_score > 0 if predicted label is in topk else 0
- alpha < 1: giving a boost to small k (top1, top2 >> top 99, top 100)
- My final submission: (max_value, topk, alpha) = (2, 200, 0.35)
```
I didn’t try many values of hyperparameters of my recognition_score, but it gave a 0.006 boost compared to the scoring scheme from the baseline kernel.
## Result
```
Best single-model score: 0.5424 - 0.5233 (resnest200, fc 512)
Best ensemble-model score: 0.5640 - 0.5354
```
## Not worked things
Here are some ideas which did not work well:
- Filter distractors by arcface head (if label not in top 200 -> non-landmark) it downgraded my score around 0.01
- Online fine-tuning with private training set, downgraded my score around 0.02

That might be a little long for now :)). I will update this post if I remember something interesting. Thank you for reading my sharing!
