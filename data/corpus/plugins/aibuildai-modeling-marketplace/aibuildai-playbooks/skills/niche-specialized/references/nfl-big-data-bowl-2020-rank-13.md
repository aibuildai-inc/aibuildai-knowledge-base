# Private 13th place journey (pocket side)

Competition: nfl-big-data-bowl-2020
Rank: #13
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/124851

This competition was one of the best competition that I can remember.  
Diverse approaches, scores never stagnating, real world setup, no leaks and interesting data :)  
So I want to thank everyone involved, including the hosts, kaggle admins, the competitors and of course, my teammates.  
  
Since I got an unexpected gold medal, I am going to write about my journey in this competition.  
By the way, our final model was a single model, best described by TripleLift (https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119331).  
  
I started this competition early, with my hopes really high.  
It was a while since I gave kaggle a serious shot, and I was prepared to commit myself to this competition.  
The competition metric is basically an RMSE for each yards, so I built a LightGBM model for each 199 yards.  
Of course, this makes runtime really long, but since only the yards -5 to 15+ really matter for the score,  
I predicted yards -5 to 15 and filled the other yards with average.  
This gave me first place at that time, and I was fooled to believe this would be a winning approach (so silly looking back, haha)  
  
After a while, I started dropping to the silver zone.  
Because making new features consistently improve my score, I decided to merge with my teammate Emin, who seemed to have a lot of good features.  
With his NN and features, we reached close to gold zone (but not enough).  
Since we were so far away from the top, I was thinking that there must be some kind of magic.  
I searched really hard, noticed the 2017/2018 difference of S and A, tried to make "the magic" happen, but couldn't make it.  
After a while, I noticed that adjusting the S and A to 2017 improved my LightGBM model, so I stuck with that adjustment.  
In hind-sight, this was really bad of me, because I should have tried different approaches for NN.  
This was a extremely important, because our after competition experiments show that adjusting to 2018 like Patrick did    
(https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314) would have given us roughly the same score (2nd place).    
    
Anyways, I was stuck in silver, and I made a last minute merge with TripleLift and yuhao.  
The merge went way better than I expected. Their model was a totally different one from ours, with basically no features and a simple embedding NN.  
I thought that this could have been all the difference, so I put in a lot of efforts to improve the NN.  
The most positive change I made was adding the "T+n" feature, which is where the players are, n seconds after the start.  
I also removed a lot of garbage features based on my experience of GBDT in this competition which might have improved the model.  
The last part was adding a multi-output. We were doing a softmax classification, so I added a regression part as well.  
This change was good in certain folds, especially the LB, so we used it, but not so sure about how positive it actually was.  
    
This ended us just below the gold zone in publicLB, which was really disappointing given all the high hopes I had.  
I didn't expect much shake of the LB because of how consistent our CV and LB was throughout the competition.  
However, In the end, I was proven wrong again, and reached a happy gold medal this time.  
  
Thanks for reading, and good luck to everyone in the future ;)
