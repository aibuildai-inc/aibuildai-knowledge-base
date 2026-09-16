# 10th Place Sharing

Competition: nfl-big-data-bowl-2020
Rank: #10
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119395

First thanks for NFL and Kaggle for such a competition. It was a journey with lots of learning for us in terms of NFL NFL domain and trying different Data Science ideas.

What we did is simply this: Try a lot and fail a lot. But find some useful at the end. Since none of us in our team were used to complex DL models, we were only using MLP and we were mostly focusing on extracting smart features and how to feed them into MLP.

We are happy with our score because we were able to get 1268 with only MLP and many features.

Let's pass on more detail:

**Model &amp; Features**
Our model is just a normal NN with 5 inputs.
2 of them are for categorical features, 
1 is for distance/speed &amp; distance/speed projection features, 
1 is for angle &amp; angle projection features,
1 is for aggregation statistics features.
.png?generation=1574945297226928&amp;alt=media)


**Validation**
We used 5 GroupKFold on ‘Week’, but actually the result is similar to grouping on ‘GameId’
We validated and early stop our model with &gt; 2017 CV only, so in stage 2 it will early stop based on 2018+2019 data.


**About Rusher’s Speed and Acceleration**
Rusher’s acceleration is useful to predict Yards in 2017 but not 2018:


Rusher Speed in 2018 is super linear with the distance but not in 2017:


Since the host said both S and A were calculated by Dis, we started to think the root cause is the location tracking. But then the host said the location tracking has been accurate. Therefore, our next guess is about the sampling frequency. The chances are the timestamp of sensors’ readings got messed up, or a resolution problem: because the sensors in 2017 and 2018 are different, their reading may have been forced to the same frequency 10 Hz. 
We don’t know the answer, but we then try the following 2 things:

1)	Rescale 2017 A and S with 2018 mean and std  -&gt; a little improvement in LB
2)	S = Dis * 10 -&gt; not improvement so we dropped it.


Ok let's talk about the magic jumps:

**Jump 1: Yards Post Processing**
Same as other teams, we corrected some impossible Yards prediction based on YardLine.

**Jump 2: Angles**


- Alpha: The relative angle between a player and the rusher
- Theta: Reverse of Alpha
- Beta: The different between Rusher’s Direction and the Theta, it indicates if a player is blocking the rusher’s path. Small abs(Beta) indicates the rusher is running towards the player.


**Jump 3: Future Features Estimation**
Features about what is going to happen, in next 0.5, 0.75, 1, 1.25, 1.5 seconds
1.	the distance between rusher and closest players.
2.	the distances between defensive players and offensive players. 
3.	Their distances to the yard line are important.
4.	Speed is also projected, the projected horizontal/vertical speed of the rusher is important.


**Jump 4: Standardizing Rusher Direction**
Consider the below 4 plays. The Black arrow indicating the rusher’s direction. We flipped all play with rusher’s direction pointing downwards. And the result is the blue arrows, after the flipping, you can see 1 &amp; 2 are the same, 3&amp;4 are the same. As a result, all rusher’s direction will belong to [270, 360] and [0, 90].

This flipping is tricky, we have to make sure the entire play is mirrored along the middle line. 



**The end**
Thanks to the host for offering this nice competition. We have learned a lot here.  Also after reviewing other talents' approaches,  we really need to update our deep learning skills.
