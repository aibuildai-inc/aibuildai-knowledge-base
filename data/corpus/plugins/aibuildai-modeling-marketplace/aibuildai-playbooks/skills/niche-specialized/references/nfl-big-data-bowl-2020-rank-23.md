# 25th Public LB, A great learning experience

Competition: nfl-big-data-bowl-2020
Rank: #23
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119461

First at all, good luck to everybody for the second stage, and thank you to all the people who with their topics, kernels and, now, solutions make this a great competition where you can learn so much.                                       






###### 
In a personal note I’m very happy, not because of the position (as I think I’m going to get worse on private or probably my submissions will crash) but because for me, as I said in the title, this has been a great learning experience. I started it knowing almost nothing about NN or python  and I end it getting a python NN submission good enough to get to a good place. I still have to learn a lot but I think is a good start.  Specially because is something that I really wanted it to do but I had been postponing as I feel quite comfortable with R and GBM models. This competition worked as a very good push to do it. It was out of necessity because as @dmitriyguller  said in the using R post:

&gt; Unfortunately my experience has been that taking part in this competition using R is like running a marathon while breathing through a straw.

Sometimes I think it would had been better if the hosts had limited the competition to python kernels, but then maybe I wouldn’t had started so I don’t know. 



###### -
With R, and a 199 multi-class LGB I was able to get to .01305 on the LB, at that moment a gold medal position, but I couldn't get better, so I started to learn about NN and Keras and almost from the beginning, using the same variables, I observed that you could get better results. The problem was that keras doesn’t work on R kernels so I had to learn python. 


###### -

My final solution, is not very different to some of the solutions already explained: most of the variables are positions and distances to the rusher in different  moments of time using S and Dir, and then some max, mins, means and sd over them. I also flip the Y and did post process to overwrite to 0 or to 1 impossible  yardages. I also engineered some features trying to capture possibles blockages of the non rusher offenders over the defenders. I feed all this variable into a simple sequential NN  (512,256,128) + dropouts(0.5) with elu activation and adam optimizer, and softmax final layer . Before that I run a Catboots regressor to predict the number of yards and use the prediction as a feature for the NN.


###### -

I spent quite a lot of time trying to get a more complicated NN to work. I got a first promising try getting the positions of the player into a grid an feeding into a CNN as it were and image, then using the positions in different moments of time and feed them into a CNN + LTSM NN as it were a video, and finally trying different combinations to get the player variables into a CNN. I had the feeling that being able to mix it was the magic to get a really good result. I think the Zoo solution proves my intuition was right but I was not even close to make it work. As I said, I still have to much to learn.

###### -

Because all of this I wasn’t able to do submissions for almost the last 15 days of the competition and I got my first working python solution two days before deadline.  The worst thing of the transition from R to python and the final rush, is that I lost the correlation between my CV scores and LB that I had when I was using R and LGBM so I’m not very confident that my final submissions do well in a possible shake out. Also my CV using python keras  is 0.00007 worse than my local R keras CV. I don’t know if is because of randomness  of folds or NN or because I didn’t translate well some variables from R to python. I probably make some errors and it wouldn't be strange if my submissions doesn’t work with the new data. It wouldn’t make me very sad as I’m very happy about the things I learn.  

###### -

This is why I want to thank again to all the people who share their insights and wisdom and make this a great community. Medals, leaderboard positions, and competition in general are very good things to push you to try harder, but the important, at the end, is the knowledge you acquire.
