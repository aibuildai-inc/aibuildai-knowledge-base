# 10th - Looney Toon

Competition: ncaam-march-mania-2021
Rank: #10
Source: https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/231068

Thank you to Kaggle & Jeff Sonas for running this competition for so many years now - I think Jeff in particular is a superstar standout competition host - reading over the past competition forums, there are a *lot* of helpful and insightful interactions.

They are fun practice & become addictive - I went overboard in 2017 [plotting everyones submissions][1] and [computing all the possible outcomes][2].

I love the originality and ideas in the early competition forums like feeding in the height of players, and distances travelled to games, but the ultimate aim for me is just to do something others have not done, to stake out some place of your own in the 'upset space'.

Because of the unique circumstances of this years tournament, I chose to re-use the [Madtown approach from 2019][3]. Check out their post, they explain it brilliantly. In summary: they do a Bayesian logistic regression with `pystan`, then flip the *middling* predictions that originally lie between 0.23 and 0.77 to be all 0.36 in one file and 0.64 in the other, so there is high divergence between submissions - approximately half the scored solution rows are changed by doing this. Depending on how those balanced games go, one submission (hopefully) will get more correct 0.64's than the other, and the nonlinearity of the log scale means overall average loss is better.

In case others were also reusing this I decided to add more randomness. Instead of using team ID to split the low-confidence predictions I tried purely random 50/50 partitions, and looked at the effects it had on log-loss and LB rank for past competitions. Other schemes might be better than just random but I didn't get around to trying. Running their model on past years then flipping those middling predictions differently 1000 times and counting the resulting LB rank for the *pair* of subs, I get:

```
Year     BestRank   p(Top11)   MedianRank
2014        1        .327          16
2015        1        .097         114
2016       37        .000         349
2017        1        .387          12
2018        1        .130          83
2019        1        .267          55
```

I also tried a simple grid search over different thresholds/constants but that just confirmed the values they used seem to be optimal. 

The median results are generally better than my actual past results and better than I expected to see; and there's an 'expected gold' probability of about 25%. It was really fun to test this code out and see the log-loss asymmetry at work, it's a quirky, nonlinear, counterintuitive (to me) trick. I felt disappointed I hadn't thought of this, but then, who did?

To answer, here are my submissions as heatmaps:

[Heatmaps]

Notice the dark (high confidence) predictions are the same for both but the lighter red/blue (flipped middling predictions) are set to 0.36 and 0.64 and swap colors in each submission.

Here are the same style of heatmaps for the past submissions that were released publicly:

 - [2015](https://www.kaggle.com/jtrotman/beautiful-mania-2015)
 - [2016](https://www.kaggle.com/jtrotman/beautiful-mania-2016)
 - [2017](https://www.kaggle.com/jtrotman/beautiful-mania-2017)

Of all the beautiful, diverse approaches on display, it seems no-one used this until 2019.

The downside is - it makes for an entry that jumps up and down the LB semi-randomly. I don't think it is a meaningful comparison to look at the score amongst other entries until the end - and that *ended* the fun - it was a bit like watching a PRNG at work... I had even *less* of an idea who to root for. I didn't look at predictions until yesterday: my leading submission's prediction for the final game was 0.36 (for Gonzaga) which is in the middling zone, so it was that way just because it had been randomly assigned, before the tournament even began, somewhere in the silicon innards of my (t)rusty old MacBook... I assumed Gonzaga would take it and I'd fall. Well done & thank you Baylor!

Every gold medal is only worth the story behind it and I've admitted this is luck - about a 3:1 shot. In football, teams now measure expected goals, and in F1 analysts can work out [expected points per race for drivers][4], I have the fairly unique distiction of knowing I'm actually on 2.25 "expected gold" medals instead of three; I trust they won't make a UI update just for this :D

I'm sorry to the teams I jumped ahead of... Whenever people say they are heartbroken to miss out on gold I think of [this essay by Richard Dawkins: The tyranny of the discontinuous mind](https://www.newstatesman.com/blogs/the-staggers/2011/12/issue-essay-line-dawkins). The gold medal line is arbitrary! Non Kagglers understand "top 2%" or "top 7%" much better anyway, saying 'gold' throws away information and requires explanation. The article also makes interesting points worth thinking about in the wider world. I hope it helps. And as I've said to myself four times now, there's always next year :-)

______________
*Edit: see the Post-Mortem comment below for an important extra detail and ... a twist*
https://www.kaggle.com/c/ncaam-march-mania-2021/discussion/231068#1267833



[1]: https://www.kaggle.com/c/march-machine-learning-mania-2017/discussion/30333
[2]: https://www.kaggle.com/c/march-machine-learning-mania-2017/discussion/30680
[3]: https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/90254
[4]: https://f1metrics.wordpress.com/2019/11/22/the-f1metrics-top-100/
