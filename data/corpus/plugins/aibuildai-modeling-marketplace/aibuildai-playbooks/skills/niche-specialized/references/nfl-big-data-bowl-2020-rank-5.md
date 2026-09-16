# Public LB 9th Place solution highlights

Competition: nfl-big-data-bowl-2020
Rank: #5
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119383

# Pitch control

Using @statsbymichaellopez VIP hint helped a lot. [Here](http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf) is the paper.

To implement it, I used [this](https://www.kaggle.com/pednt9/vip-hint-coded) good kernel (thank you @pednt9) and made it run under 10ms per play for 224 points in front of rusher.

I did not keep track of all improvements on LB, but to be safe, this gave me between 0.00010 and 0.00030 boost.

# Tabular model

My model is a derivation of the standard TabularModel from Fastai with several submodules as described here:

.svg?generation=1574941608426908&amp;alt=media)

# Loss

My loss activation is [softplus](https://pytorch.org/docs/stable/nn.html#softplus), then I normalize the output so it sums to 1 and backpropagate CRPS.

```
inps = softplus(inps)
inps = (inps / inps.sum(1).unsqueeze(-1)).cumsum(1)
return (inps - targ).pow(2).mean()
```

# Rusher heading north-east

A lot of competitors flipped plays so they all happen left to right. I also flipped them so all rusher are heading with increasing Y. The latter gave me a 0.00010 LB boost.

# Biggest regret

Not cleaning S nor A as @wimwim explains in his [solution overview](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314#latest-683357). I guess this is a beginner mistake to not analyse initial features well enough and not catch inconsistencies. Lesson learned !
