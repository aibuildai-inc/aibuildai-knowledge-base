# 1st solution with code(cv:0.748 lb:0.742)

Competition: feedback-prize-2021
Rank: #1
Source: https://www.kaggle.com/c/feedback-prize-2021/discussion/313177

First of all, I would like to thank the competition organizers for this great competition. At the same time, I am very grateful to many brothers for providing very good notebooks and discussions. I learned a lot from them and applied them to our final solution. 

In this competition, because each token needs to be combined into a sentence as the final result, in order to reduce post-processing, we divide our solution into two stages

# stage 1: bert token prediction
First thanks to the excellent notebooks and discussions @cdeotte @abhishek @hengck23 and others
https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633 
https://www.kaggle.com/abhishek/two-longformers-are-better-than-1
https://www.kaggle.com/hengck23/1-birdformer-1-longformer-one-fold
https://www.kaggle.com/c/feedback-prize-2021/discussion/297591
https://www.kaggle.com/c/feedback-prize-2021/discussion/308992
etc...

we tried various pretrain models, since the max length of some models is 512, for these models, we choose the method of segmented prediction and splicing, Finally we choice longformer-large,  roberta-large, deberta-xxlarge, distilbart_mnli_12_9, bart_large_finetuned_squadv1 for ensemble. This stage takes 7 hours online, and **cv score 0.712**, **lb score 0.706** with post-processing.

here is each pretrain model cv score:
[[1647309894393-e37bd0b0-7aac-45e6-a7d0-d749e3a10fb2.png]](https://postimg.cc/c6hTg2qd)
We put the bert training code [here](https://www.kaggle.com/wht1996/feedback-nn-train), Because the kaggle online resources are insufficient, you need to copy it to your own machine for training

# stage 2: lgb sentence prediction
Thanks @chasembowers for the excellent notebook https://www.kaggle.com/chasembowers/sequence-postprocessing-v2-67-lb

we first recall as many candidate samples as possible by lowering the threshold. On the training set, we recall three million samples to achieve a mean of 95% of recalls, the recalls for each class are 
| class | recall  |
| --- | --- |
| Claim | 0.938 |
| Concluding Statement | 0.972 |
| Counterclaim | 0.906 |
| Evidence | 0.974 |
| Lead | 0.970 |
| Position | 0.928 |
| Rebuttal | 0.895 |

in addition, after getting the recall samples, we select sample with high boundary threshold and choice 65% length with the highest probability of the current class as a new sample, this method can help improve score about 0.008. Finally, We made about 170 features for lgb training, and select some samples as the final submission. This stage takes 1.5 hours online, and **cv score 0.748**, **lb score 0.742**.

We tested our lgb on the 5 fold longformer model, and the score increased from [0.697](https://www.kaggle.com/wht1996/feedback-longformer-5fold-0-697) to [0.727](https://www.kaggle.com/wht1996/feedback-two-stage-lb0-727). Because lgb is not trained on this prediction, the improvement will be lower than the actual. At the same time, we uploaded our model ensemble results [here](https://www.kaggle.com/wht1996/feedback-two-stage-cv748). If you are interested, you can replace the prediction results with your own to see how much the cv score can improve.

# Summarize
**useful attempt:**
1、adversarial learning (awp/fgm): cv increase 0.01，lb 5fold ensemble increase 0.003.
2、model ensemble:  single model lb 0.691, model ensemble 0.706, longformer and deberta ensemble increase most.
3、lgb sentence prediction: cv increase 0.036, lb increase 0.036,  among this, select sample with high boundary threshold and choice 65% length with the highest probability of the current class as a new sample can increase 0.008

**useless attempt:**
1、add paragraph information to input
2、back translation
3、adjust the weight according to the position of the sentence in which the word is located
4、lgb model use overlap percentage as label to train
5、stage 2 use bert to predict and ensemble with lgb

**code:**
longformer train: https://www.kaggle.com/wht1996/feedback-nn-train
lgb train: https://www.kaggle.com/wht1996/feedback-lgb-train
5fold longfomer with post-processing(lb 0.697): https://www.kaggle.com/wht1996/feedback-longformer-5fold-0-697
5fold longfomer with lgb(lb 0.727): https://www.kaggle.com/wht1996/feedback-two-stage-lb0-727
cv ensemble with lgb(cv 0.747): https://www.kaggle.com/wht1996/feedback-two-stage-cv0-747

Our code and data is published on GitHub [here](https://github.com/antmachineintelligence/Feedback_1st)
