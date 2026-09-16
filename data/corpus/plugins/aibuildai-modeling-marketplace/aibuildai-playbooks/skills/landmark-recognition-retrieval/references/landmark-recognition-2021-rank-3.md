# 3rd place solution

Competition: landmark-recognition-2021
Rank: #3
Source: https://www.kaggle.com/c/landmark-recognition-2021/discussion/275870

## 3rd  place solution 
Congrats to the winners, and thanks a lot to Kaggle & Google for organising this wonderful Landmark Recognition competition yearly. This is great a teamwork for us, and we will briefly discuss our solution here. 

We are probably the most submitted team (200+ submissions) on LB, reason being we were exploring on different approaches quite independently in the beginning, where we only merged in the final week right before the deadline. We have explored quite many different ideas from the start, but our final ensemble pipeline differed a lot from what tried in the beginning. 

Probably like most others, we wasted quite a few submissions before realising that 1) there are at least 20K private test images to be provided during inference, instead of 10K as in test folder 2) there are landmark ids other than the cleaned 81313 version existing in private train.  

### Solution Overview

Our final models include two sets - **retrieval models** and **classification models**. And our solution can be summarised the following: 
`Concat & Retrieval -> 
classification logits adjustment -> 
distractor score adjustment -> 
top 1 classification score `

### Dive into details for each part

####1) Concat & Retrieval 
We have 7 retrieval models in our final pipeline: 

OrangeX: Swin-l (384), Swin-l (384 , different training), EfficientnetV2 (800), EfficientB7 (800), CvT (384)
Weimin: EfficientB5 768

We found that training models only on 3M (81313 landmarks) dataset gives better or at least equal retrieval performance compared to finetuning on full 4M datasets. And our single best retrieval model - Swin-l - gives 0.4 for its pure retrieval performance on Public LB. All others like B7, V2, B5 are a lot weaker in terms of individual retrieval performance. (Around 0.29 ~ 0.33). 

All our models give 512-d output features, and in our final pipeline we simply concat them to form a final embedding vector (512 x 6) for retrieval of top K matched training images from private train. We used K = 7. 

We found that using dynamic margin (or adaptive margins) actually helps improve training, thanks To Qishen's great solution last year! 

####2) classification logits adjustment
We found that it is crucial to use classification logits to support predictions. From the beginning, I managed to split the full 4M dataset into training and validation folds, and when I realised the private train could contain un-cleaned landmark ids, the validation mean_gap can be reliably used as measurement for classification performance, which is also consistent to LB. 

It turned out that this is important. As using a representative validation set, it is relatively easy to finetune models back on full 200k landmark ids training set (4M images). In the end, we found the top 1 pure classification accuracy of my B5 is around 0.39+ on LB, where for swin it is around 0.34.  We decided to go with all 4 EfficientNet that I trained on my side (B5 512 & 768, B6 512 & 768) as our main classification models for stage 2) here as well as 4) later on. 

At this stage 2), we have got top 7 retrieved training images from stage 1). So for each of the 7 images, we look up its classification logits from all 4 models chosen (B5 512&768, B6 512&768), and simply add the averaged logit to its corresponding cosine score as adjustment. 

####3) distractor score adjustment
Similarly from what Dieter did last year, we use the 2019 test set's nonlandmark images as index, and for each training image (4M), we found its top 3 matched scores and simply take its average as its distractor score. We generated the mapping between each 4M training image id to its score in a dict, and uploaded to Kaggle to use in submission. 

We then subtract the distractor score from each adjusted cosine score above from stage 2). 
In short, stage 1) - 3) can be viewed as: 

```
Cosine score + classification logit - distractor score 

```
####4) top 1 classification score 
We found that using the top 1 classification logits from our best classification models (i.e. EffNet B5 and B6) can have another boost. 

At stage 3) above, we should have all top 7 indexed images with their score adjusted ready,  so we simply aggregate them to each's corresponding landmark id. We will, however, add another pair of `(top 1 classification landmark id, top 1 classification logit)` into the aggregation step. The classification logit used is just the raw top 1 logit, after we averaged all classification models' 200k prediction logits. 

We can't penalise the classification pair as it is not from any image like the top 7 matched, therefore it does not have a distractor score. But this turns out not to be a problem, as we found that top 1 classification score can be naturally used as penalty of non-landmark. See the distribution of landmark & non-landmark images' top1 scores below: 

https://drive.google.com/file/d/127O__NgWGIW8E73XX2ZqrK4X9m9asZY_/view?usp=sharing

Our final selection of landmark id and score for each test image will be just the aggregation result from all 1) - 4) stages. 

Using stage 1) - stage 4) with a single B5 only (not fully trained yet) achieves 0.445/476 on LB. We achieved our final rankings using all models mentioned above.
