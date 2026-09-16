# 2nd place solution & private LB 0.577 QUICK OVERVIEW

Competition: novozymes-enzyme-stability-prediction
Rank: #2
Source: https://www.kaggle.com/c/novozymes-enzyme-stability-prediction/discussion/376134

Hi everyone! My solution is very simple. It's just an ensemble of different methods and a little smoothing at the end. 

a quick overview:


In the +ve weights block we have different methods. let me know if you want the explanation of each of them separately, I will make another post for it but as they are available on the internet, I didn't write about them here.

I'm assuming that up to the ensemble, everything is clear.

Now coming to the smoothing part:

for me, combing ML models (the ones I have mentioned) with the positive block drops the score, my assumption is that it makes the values extreme. even if we overfit the public LB, we might drop the private LB. 
Private LB was mostly from ~500 rows to ~1600 rows so if we have extreme values based on public LB, this part of private LB could drop significantly. So to avoid shakeup we needed some smoothing.
Now we have many methods for smoothing, we could simply try log transform but the problem is we don't know which part of the prediction needs smoothing and which doesn't. one method to solve this is to train a neural network but as we have training data belonging to different distributions, this won't help us here. 
So I did something risky: I ensembled all the ML models and subtracted them from the ensemble.
let's say we have an ensemble prediction of 0.9 and ground truth of 0.5 and an ML prediction of 0.6. now subtracting ML prediction from ensemble 0.9-0.6 = 0.3 will put us a little closer to 0.5.
The reason I called it risky is that if you didn't set the right weights, you will fall terribly on private LB. assume ground truth and ensemble both 0.9 and ML pred as 0.6. it will add a big error difference here.
So to make it work, we need to set weights based on CV scores but in this competition, we don't have similar training data so I had to set the weights based on public LB. 

With different combinations of the above, I was able to make about ~30 submissions in the range of private scores 0.545 to 0.577. I found their private and public score a little reliable and correlating.
here is a snap:
.png?generation=1672863185750539&alt=media)

This was the first time I was working on a protein-related competition, I tried many different things everyday for the past 3 months (80% of them are not part of my solution) and I'm really happy that I won a solo gold. I hope to become GM soon.
Thanks.
