# Solution and some ideas

Competition: data-science-bowl-2019
Rank: #42
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127306

First of all, congrats to all winners! Was an interesting competition in the end, with not too much signal in the data and a very weirdly behaving metric. This is not a gold medal solution, maybe a few points are still interesting to some of you.

I joined the competition only a bit more than two weeks before end without any expectations. The main reason to jump in, was that I had a few ideas I wanted to check out on data like this using neural networks which is also what I started with.

**Neural network**

In NFL competition I learned that you can use convolution neural networks really well on non-temporal data by just using kernels with size 1. So my idea was to use it as a form of feature engineering / understanding without me spending too much time on the engineering part myself. 

I observed that a few things are important for predicting the success here. The two most important things are using information from previous assessments and previous other activities. So I generated two forms of sequences for each training data: sequence of all previous sessions, and sequence of all previous assessments. Each step in a sequence can then have multiple features, like one-hot-encoded or embedded title. So the assessment sequence could look like: assessment title 1, assessment title 2, etc.

After padding and reshaping, I tried to run LSTM and CNN on top, but quite quickly saw that there is little temporal information, so I just used CNN with kernel size 2 in the end for the session sequence, and kernel size 1 for the assessment sequence. I had as additional input the current assessment title.

After a bit of tuning, this NN scored around 0.520 on public LB without any threshold tuning. I always only did fast submission though and only used the training data externally. And then I probably made a mistake. 

I decided to use my evaluation routine (more on that a bit later) on one of the public kernels to test it out. I also was fixed on the idea to use all the data available from test to train on, so I definitely wanted to train my model in kernel. The public kernel then quickly scored 0.557 with me adding my routine and the extra data. So I thought: wow I have a good setup let me try to improve that. So I decided to not port the NN to the kernel, because my code was very memory heavy and I would have needed to work a few days on adjusting it and also check how it works with training on more data. So from this point on, I do not use this NN anylonger, but I still believe it has potential if properly tuned and adjusted. What I also want to mention is that the public kernels had a lot of bugs, and it took me quite some time to find most of them. I think it is way better to start from scratch the next time. It is the first time I started with some public kernels.

**CV and thresholds**

I believe I came up with a quite nice and robust CV setup including a nice way to optimize thresholds. So what I did was to use stratified group kfold, and final CV is based on the median score of a few thousand truncated samples. I believe a few evaluated their models like that. What I did with thresholds though, was to **optimize the thresholds in a way that they optimize the median score of these truncated samples**. I then used these thresholds to predict the test set. To improve the threshold optimization I initialized the Nelder Mead algorithm with the histogram of the target.

**Features and models**

As said earlier, I don't think there is a tremendous amount of signal in the data. So I did not spend too much time on FE, even though I believe that some carefully crafted features can help quite a bit. In the end I used event codes, event ids, a few assessment related features and a handful extracted from event data. I tried other things like tfidf on json data etc. without too much success. I train on **all samples**, meaning also those extracted from test data.

I focused on LGB and Catboost in the end. With catboost I explicitely utilize the `has_time` parameter which is perfect for this competition as it encoded categorical variables based on time information it has, so only using samples before that timepoint. I explicitely added also the `assessment_id` as categorical variable as local tests suggested that it will help me on private LB if I add the previous records to training.

**Blending**

I tried to be as robust as possible, so I decided on the following schema. 15 times 10 fold, for each of those 15 bags fit catboost and lgb, blend them with rank average using catboost 25% and lgb 75%. Optimize truncated thresholds as above. Predict test based also on rank avg mean of all 10 folds and then finally do majority voting on all 15 bags. I am quite happy with that because I managed to pick one of my best private LB scores in the end.

**Crazy ideas that did not make it**

I had two "crazy" ideas. I think the first one is quite simple and should have worked, but I had a bug on the kernel at the last day and didn't select it even though it scored highest on private LB even with this bug. The idea is quite simple, fit a MinMaxScaler or QuantileTransformer on the  test predictions, transform oof and test predictions with it. Then do the threshold optimization on oof, and then apply it to test. This brings them on a similar range and scale and has some benefits over ranking the predictions. In nearly all my local experiments with simulating test data, this improved the QWK on the test data. I think this can bring a few points, but I have to test it again.

The second idea involves again CNNs. The idea is to use CNNs to find the optimal thresholds for a sample. My idea was to do repeated subsampling of predictions on oof (can also be truncated) and calculate the optimal thresholds for these subsamples. The predictions of these subsamples are then the training data, and the thresholds the targets to predict. But again, you have no order on the training data but rather a set of predictions. So you can take a 1 kernel CNN and fit it on the set in order to predict the three thresholds optimizing something like MSE. Evaluation is then how close predicted thresholds are to the optimal thresholds. This actually also worked really well locally, but again I did not have time to tune it properly and port it to kernels. But might be worth a shot to test this further.

I tried really hard to find a good gamble the last few days using local test simulations, but I just could not find any. Ideas like sampling according to the test distribution etc. 

This competition again is a good example of why you should not chase the public LB. I was also tempted to do it after jumping to position 50 or so, but relying on a robust CV setup simulating how test data looks like is usually the better idea.

Kernel: https://www.kaggle.com/philippsinger/lgb-catboost-17-10fold-15bag-3k-majority-rank-v4?
