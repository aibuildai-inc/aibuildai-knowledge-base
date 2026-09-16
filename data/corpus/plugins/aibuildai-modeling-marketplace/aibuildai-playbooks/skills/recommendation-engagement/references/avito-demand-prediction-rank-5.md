# 5th place solution

Competition: avito-demand-prediction
Rank: #5
Source: https://www.kaggle.com/c/avito-demand-prediction/discussion/59914

Congrats to top3 winners, and all top teams – its funny to see the solutions – I was pretty sure there was something massive we missed on images, but after a first read of other solutions it looks like there was no big thing missed there.   
When we got into the competition we would have been pretty happy with a top50, so are psyched to end up with 5th. Our team are all working together at Optum Health (hence the name :) ) so it was great to benchmark some of our techniques here.    
Our solution involved a LGBM stack of 4 different types of models – lgbm, RNN, MLP and ridge. The best of each scored on public LB approx `0.216, 0.2185, 0.2215 and 0.222` – but it was really in the stack where diversity between these helped. This is our repo [linky][1]. We tracked progression for different models/changes in there on the front page.    
For teams starting in Kaggle or Data Science I cannot underestimate the importance of getting a good local validation that tracks to the Leaderboard and tracking improvement on val and lb. Initially we used a small validation set of a few of the final days of train; when we had a few good models we set up a stack and we used 5 fold with timesplit.    
 
**MLP**    
In general we leaned a lot on the Mercari solution’s it was a very similar problem. The winning MLP there scored 0.2215 here with very little changes ( [MLP code][2] ). Tried a few other things to improve, but it did not help in the stack. All credits to Konstantin and Pawel who developed this and for sharing a simple 75 line version [linky][3]      

**RNN**    
Tim set up the RNN on a macbook GPU which was pretty impressive . We also introduced pymorphy2 which gave good improvement here and on lgb for tokenization. A lot of work was done on regularization for tuning and we concatenated on the penultimate layer of a densenet feature map of the images.  Example [RNN code][4]    
We tried adding numerical features and pretrained embeddings; while it helped at L1, it did not add much in the stack for the first few tries. One of the main challenges here was hardware – on 5CV it took about 36 hours to run on an AWS P2, as we bagged 2 times and used 256 wide embedding layer – so we gave up here and concentrated a bit more on LGB.   

**LGB**   
Features engineering in LGB was the happy tree that did not stop giving. On the repo front page you can see the progression.    
One of the strongest was relative price. We did a kind of Bayesian mean of item price vs price of the group – `((item_price/mean_price_grp)*ct_grp + (prior))/(ct_grp+prior)`. This allowed the ratio be weighted on how many items were in the group - which is pretty important, if an item is the cheapest of 2 similar items its a lot less significant than being the cheapest of 100 similar items.  Just doing this over lots of different groups – title, params categories, clusters etc. added close to 0.002.    
This is an [example][5] of price ratio.   
Image features in the public kernels helped a little – dullness, channel intensity etc. All credits to the author.     
Bayesian mean and counts over different groups helped maybe 0.001 also. We used a few combinations of tfidf for diversity. Entropy helped also. Dropping categoricals in their raw encoded form helped; and letting the model learn their representation through the FE mentioned. Also moving up to 1000 leaves helped some – 2000 or 5000 leaves probably would have helped more, but took too long to run.   

**Ridge**    
Late in the game we set up a few ridge models; they were very fast. Although a lot weaker, they gave about 0.0005 on the stack. We had a separate model running on each parent category; models on image feature maps (vgg19 and densenet) – and on different types of count vectorizer and tfidf of text features.   

**Stack**   
The final stack just combined about 30 different models – also added all two way combinations of sums and differences of each model. We got an MLP stack scoring approximately the same but correlation to the lgb was very high so averaging did not help.   

**Final note**  
Ok, I was wondering this morning how teams have such long write-ups, so looking back I can see why. There was so much opportunity to try things – thanks to Avito for hosting this and I hope the Avito team get some benefit from the solutions and we’ll see you back on Kaggle with another competition soon! 


  [1]: https://github.com/darraghdog/avito-demand/
  [2]: https://github.com/darraghdog/avito-demand/blob/master/nnet/mlp_1705.py
  [3]: https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
  [4]: https://github.com/darraghdog/avito-demand/blob/master/nnet/rnntmp/nnetdh5CV_2705A.py
  [5]: https://github.com/darraghdog/avito-demand/blob/master/features/code/pratioFestivitiesR1206.R
