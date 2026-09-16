# Customed loss(10th solution)

Competition: jigsaw-toxic-severity-rating
Rank: #10
Source: https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306373

There are many no-working things before I come to my final solution, the list could build an unbalanced dataset together with the working things😄
I always believe in simpleness. Everystep should be taken with a reason or never takes it.
#####Modeling
In the end, I ensembed 7 of the variation models along:
**3 pretrain encoder** roberta-base, roberta tweet sentiment, roberta tweet offensive
**3 head** bi-lstm with layernorm, robertaforclassification, robertaforclassification+sigmoid
**4 dataset** jigsaw toxic, ruddit, davidson, olid
**2 loss** customed loss and bce loss
**seeds** when the dataset keeps , change the seed. When the model keeps, change the seed. When loss keeps, don't change the seed.
scores are summed simplely.
*There is one submission without the oild dataset could score 5th. So the olid is not good*
##### Customed loss
The loss is something like a margin rank loss. Just adapt it to be batch-wised. 
Sort the batch samples by label score ascendingly. Sort the output in the same order. And calculate the rank loss between adjacent samples. When the batch is N samples, there are N-1 pairs is summed and averaged.
As well as customed loss, it benefits to have bigger batch size. I train with tpu 8 core, the batch size is 48*8=384. I kinds of assume that a best tranning is the one without strange lr schedulers, use a constant 1e-4.
##### Trust cv
As soon as I realized the public lb is a misleading one. I closed my eyes to it. I use both validation data and its voting version with 1-fold. And remove leaky comments in toxic data.
