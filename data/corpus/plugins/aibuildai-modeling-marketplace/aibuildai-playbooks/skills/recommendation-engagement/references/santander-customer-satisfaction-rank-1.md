# Congrats!

Competition: santander-customer-satisfaction
Rank: #1
Source: https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20647#118259

Thanks to kaggle and Santander to hold such a challenging and interesting competition! And congrats to other winners, especially the #1 Leustagos! You did a great job!

On our side, the key for winning this competition is to try our best to deal with the overfitting issue carefully. I have an clear impression very early in this competition that, in this particular competition, it seems that neither the local CV nor the public LB would be reliable enough, no matter how sophisticated you did the testing. 

Some strategies we used to reduce overfitting are as follows: 1) use both local CV and public LB test to identify a couple of potentially good model candidates; 2) in our final solution, mainly use those model candidates which give "quite large" improvements and ignore those models that only provide small improvements, since those tiny improvements might be quite possible due to overfiting to the noise. For example, our best public LB score ensemble include >=60 individual models, and most of them only provide about <0.000050 tiny improvement.  In our final ensemble which gave us #1 place on the private LB, I kicked out most of these models with tiny improvements, and the final ensemble only include about 5 models (here some of these 5 models are an average of running a model with many different random seeds); 3) for xgb models, run it with a couple of different seeds and slightly different parameters, and then take the average, this will also make its performance a bit more stable. 4) For  those features like age<23 to identify 0 records, only used them very conservatively.

Btw, we also didn't choose our best public LB solution as one of our final two private LB submissions. The submission which gave us #1 on private LB is a very conservative one and it only scored 0.8429 on the public LB. And we do have an ensemble which probably would give us top3 position on the public LB (it's basically just adding some more "identify 0 records features" on top of our previous best public LB ensemble, and those features are proved to be helpful by public script for improving public LB score.) But I do highly doubt that they would be probably overfitting to the public LB and would perform bad on private LB, and thus I don't even bother to submit that ensemble.

I am a bit busy these days, but I would work together with my teammate Medrr to prepare a more detailed description of our winning approach in the following couple of days.

Have a great day!

Best regards,

Shize
