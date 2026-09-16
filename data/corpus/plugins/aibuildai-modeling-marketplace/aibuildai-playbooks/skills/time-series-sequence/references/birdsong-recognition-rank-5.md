# 5th place solution

Competition: birdsong-recognition
Rank: #5
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183300

A great journey, thanks for organizing it! Now every time I walk around I hear much more birds :) 
I've started my machine learning journey with speech processing few years ago and that was a joy to play again with it. One more time to thank and mention @hidehisaarai1213 who led us through the darkness here.

From the very scratch I understood that the problem can be decomposed into two main problems:

- a domain shift (which is pretty tough us we don't have target test set)
- a clipwise to framewise classification transit (which also introduced us additional "nocall" class with huge influence on results, and labels noise)

Tackling those two needed a validation/test set as close to target distribution as possible in terms of snr and nocall distributions. 

**Validation**

Believe me or not but with 6 records from [here](https://www.kaggle.com/c/birdsong-recognition/discussion/158877#911336) and 2 example audios I was able to get some correlation with LB even though it contained only 500-800 clips. But more important I was able to control % of "nocall"s and pick the right threshold based on them. Last two days I splitted those clips into 3 test sets with 50%, 60%, 80% "nocalls" to see possible scenarios on private leaderboard (thanks organizers private is almost equal to public) and to secure the score with my second submission. 

As a target metric I've used F1 by `average="samples"` pointed by @cpmp. By the end I've also monitored validation (default CV split) primary+secondary F1 score as my another decision-making metric. Primary F1 or mAP was not enough.



Adding separate nocall class didn't work for me for whatever reason.
And yeah, I trusted leaderboard too.

**Domain shift**
Assuming we don't have the transit problem, this might be solved simply with augmentations. I've spent too much time on this and right now I understand that was not productive. My final models contain different augmentations configurations: 
- backgrounds (all from external data thread)
- pink and brown noise
- pitch shift
- low pass filtering
- spec augments (time and frequency masking)

As organizers informed that we can train on two test examples, I tried to collect batch norm statistics from those as a domain adaptation technique but it didn't work great. 

**Clipwise > framewise transit p.1**

That's my favorite part!
So what we have here - every time we crop 5 seconds clip we have a chance to crop a nocall clip. So labels become really noisy. Even more it's hard to crop secondary classes. Easy way to tackle this? Label smoothing 0.2. I don't remember when I became a fan of label smoothing but it works well with noisy labels on practice. But that's not serious.

People here tackled this task having an energy based cuts. And it really worked. So I've tried both approaches: soft and hard. By soft I mean random sampling based on energy, by hard - removing everything below normalized energy threshold.

But what if we can use our model to extract these labels? Having avg/map pooling head gives a chance to get a free segmentation:

which you can use as soft labels or hard binarized labels. These methods showed really cool performance for primary labels. For secondary I kept label smoothing :)

**Clipwise > framewise transit p.2 (The most important part)**

I found SED models comparably weak trained on 5 sec clips but as I trained them on longer clips (10-30 seconds) I noticed that due to labels noise reduction they show much better training loss/mAP. Moreover, it was possible to run inference on 5 sec clips with nice performance. So I took B4 EffNet, added the same Attention decision making head and ... failed!

I've spent a lot on this one. So why did Cnn14_DecisionLevelAtt trained on 30 sec clips work well on 5 sec clips? Playing with EffNet I found that problem was in receptive field - it's too big (630 compared to 200 or so). With wide receptive field attention head was not able to build a meaningful framewise feature maps. As I've changed EffNet kernels from 5x5 to 3x3 or reduced number of blocks - it solved problem but price was too big - pretrained weights. 

So what we have: 
- longer input
- clipping based on some weakly labeled probabilities
- label smoothing for secondary

Having these I've decided to simply run 5sec classification inference with no sophisticated postprocessing that might fail on private LB. As it always happens, I've found this too late so trained only two PANNs models (resnet38 and cnn14 mentioned above) on 12 and 15 second clips (128 mels) with mixup and didn't have time to configure them properly.

**Finally**

I've optimized inference to run all my models (cnn14, resnet38 and few effnets) in 20 mins with `kaiser_fast` resampling and around one hour with `kaiser_best`. I've picked thresholds based on my test set (0.3) and some skepticism about it (0.4). I found I have 670+ private score submissions from two weeks ago, which I can't explain. Probably, Kaggle leaderboard is another stochastic process.

This has been a long journey in which I not only solved many competitions and learned just hundreds of tools and tricks, but also met wonderful minds from all over the world and found job of my dreams. I wish to meet you in person at Kaggle Days once covid is over. See you!
