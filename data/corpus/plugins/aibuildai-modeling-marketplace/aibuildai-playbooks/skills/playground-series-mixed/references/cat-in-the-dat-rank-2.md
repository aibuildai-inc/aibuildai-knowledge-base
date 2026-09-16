# 2nd place solution

Competition: cat-in-the-dat
Rank: #2
Source: https://www.kaggle.com/c/cat-in-the-dat/discussion/121063

Hello,

First I would like to thank Kaggle for this interesting challenge, where a regression technic was able to win.
I would like to thank all players. I would like to thank especially Robert Stockton : his post <a href="https://www.kaggle.com/c/cat-in-the-dat/discussion/113726#latest-687840">"To reach the top ranks..."</a> helped me a lot, and convinced me to look for simple things and try as simple technics as possible.

<h1>Microsoft Malware Prediction</h1>
One year ago, after reading <b>Introduction to Machine Learning with Python</b> (Sarah Guido, Andreas Mûller O'Reilly), I tried my first real challenge (<b>Microsoft Malware Prediction</b>).
I ended 10th with a blend of two LGBM models. There was a big shake up (I jumped from 170th on the public LB to 10th on private). Was I lucky ? 
But during the Microsoft Malware Prediction challenge, I learned a very important thing : <b>to trust my CV, and only trust my CV</b>.

<h1>Categorical Challenge - at the beginning</h1>
I do not have time to try a lot of challenges and to test many things. At the beginning of this Categorical Challenge, several players wrote <b>Logistic</b> should be a good solution. Great, a challenge for me&nbsp;!
Cramer's V between each feature were very low, so features were independant and Trees technics wouldn't work very well. And <a>adversarial validation</a> would be useless.

Like many of you, I began by trying one hot encoders, then target encoders, thermometer encoder, but nothing other than one hot encoders was significant.

<h1>Parsimony</h1>
Regression technics love parsimony. After trying the linear relation between target and each ordinal feature like suggested by <a href="https://www.kaggle.com/c/cat-in-the-dat/discussion/113726#latest-687840">"To reach the top ranks..."</a>, and jump ahead with the first 0.8085 score (I was lucky and proud), I tried many things to have a more parcimonous model : I tested linear relations with other nominal features, I tried to make clusters of values of coefficients of a Logistic, I tried to make bins of values, etc. 
I’ve seen there was a symetry in target mean of <i>day</i> values. 
<h1>Cross Validation / target leakage / Public Leaderboard</h1>
The difference between to top ranks on AUC was at the 5th decimal, so I needed a way to be sure to make good decisions without using the public LB (I don’t trust public LB – to trust Public LB is just like to trust only one test fold). 
And also, I stuck to target leakage ; AUC on my CV grew, but I realised two weeks ago that my public score did not.
So I had to find a way to make good decisions and to avoid target leakage too: I read the easiest way to avoid target leakage is cross validation. 
So I've repeated some cross validations on other folds than my first CV, and repeated again : before to make a decision, if I had the same conclusion with several combinations of 5 folds, the conclusion is good. Else it is not.
<h1>Final submissions</h1>
My two private submissions scored only 0.80839 (yesterday evening at 11 PM UTC) and 0.80845 on the public leaderboard : they both score 0.80282 on private leaderboard.
Yesterday at 11PM UTC, I was not very optimstic. But I saw that many of the best players submited a lot of solutions, so I had a chance, maybe they trusted too much the Public LB instead of their own CV.

My best solution on public LB (0.80852) score only 0.80279 on private.

The <a href="https://www.kaggle.com/c/cat-in-the-dat/discussion/117369#latest-680216">ordinal feature</a> trick to reach the top of the public LB was hidden in one of my post two weeks ago. Sorry ! I believe we all read new notebooks, but we are only a few to read discussions. <a href="https://www.kaggle.com/adaubas/2nd-place-solution-categorical-fe-callenge">Here is my solution</a>.

I had other solutions which score 0.80847 or 0.80845 on public leaderboard : I did not choose them, because I did not do my double ou tripler cross validation on them, or because when I did it, I had not the same conclusion as on my first CV. I didn't trust them.

I had a lot of fun, I learned a lot during this challenge.
