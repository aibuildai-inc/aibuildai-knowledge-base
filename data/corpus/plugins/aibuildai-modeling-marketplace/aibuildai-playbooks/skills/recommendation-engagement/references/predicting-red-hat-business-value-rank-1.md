# long story of #1 solution

Competition: predicting-red-hat-business-value
Rank: #1
Source: https://www.kaggle.com/c/predicting-red-hat-business-value/discussion/23786

Hello guys,

I‘m going to write a story how I did it, because knowing why I did it is hard to tell in few words:)

This competition started as a goal to reach top10 for grandmaster title, which ended up with a stressful race for top1 finalle. Luckily, I made some good decisions in the last day which guaranteed me top1 spot. For me this competition was not something I enjoyed very much but still glad everything worked out in the end. My final model is quite simple and if no leak was present, it might have been production friendly.
First, proper cross-validation set was very important. Tricks I did to have a representative CV set:

1)	Remove observations of group_1=17304, both from train set and test set; That correspons to 30% of training data set, and this group has all outcome=0 (which was used an override rule in making prediction files)

2)	Use Distinct operator for gruop_1‘s which have 3000+ number of rows (this step was very important to remove potential auc bias for several of my CV folds)

3)	Create random unstratified 5-fold cv set based on people file 



My modeling concept was rather simple – reduce original problem to few smaller ones, and combine them in 2nd level model. I have built several models using principles below:

a)	Select an activity in each group_1‘s timeline (I used first/last activity in a timeline)

b)	Collect all other activities within a group which has the same outcome label

c)	Aggregate features – tf-idf was especially useful (in short, for each varriables‘ attribute calculation was done: #of people with same attribute in a group/#of people with same attribute in population)

d)	Add other simple and not so simple features (i.e. group_1 id value, #activities in a group, #people in a group, min/max dates etc.); I did not use any feature interactions or likelihood features.

e)	Build a classifier on the dataset – I only used xgboost; was able to reach ~0.84 AUC (bare in mind, no leakage has been used till this point yet!) which in my mind was a fantastic result; If I was RedHat, I would have made this as a target to model, but oh, nevermind.

To do it properly I had to think of some novel cross-validation approach as my split was based on people_id, and on this aggregated data level my CV had to be based on some kind of aggregated CV split scheme as well; My approach worked well, but along the way I had to build ~15 xgboost models to make it work (this was necessary to make 2nd level model work). Such proper CV scheme was important, but I‘m not going into details how it works for now. 

So at this point I have 4 very well performing 1st level models (2 models which perform very well on public LB, and 2 simillar versions which yield best CV score in 2nd level model); 2nd layer model required very careful scripting skills, as I implemented leakge solution in a cross-validated way, so the predictions of outcome changes within group would be learned in a ML way (simple rules in public scripts are not that good!); So in fact 2nd level model solves 2 problems – predict probabilities imputing observations affected by leakage, and predict probabilities for observations not affected by leakage. The model itself is simple but some smart features were included as well to capture time trends within group/population.

So at this point I have very decent model, and in the middle of competition I see myself at #4 place, and I see top3 guys improving their score day by day. So I put some time into it to think about it & discover that they have been doing some manual exploitation using public LB to increase the leakage. As public/private split was random (took some time to discover), one can use a hand-crafted submissions to test groups not affected by leakage to get auc results for that specific gruop, and determine what the probable outcome of whole group is. Do this for as many submissions taking largest group_1‘s as you can, and you might detect some gruops, that ML model misclassifies badly; having that in mind manually create some overrides based on that; So in my submissions i started using rule-based overrides to corret ML model shortcomings.

For my final submission i used a simple average of my best performing LB model and best performing model on CV – this was my last effort to overtake Victor, and for my surprise it provided very high uplift to my score;  I want to thank Victor for not making this easy for me in last few days, as I got myself a little too relaxed at one point:)

p.s. if I did not used any overrides I think I still would have finished within top 5. Sorry for NoHat team, which seems to have made the best ML model there.
