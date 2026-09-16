# Solution sharing....

Competition: santas-uncertain-bags
Rank: #7
Source: https://www.kaggle.com/c/santas-uncertain-bags/discussion/28286

So how did you do it?

There were some really impressive scores and leaderboard improvements  towards the end of the competition that lead me to conclude that the winning strategy wasn't quiet so obvious after all!

I'd really like to know what Bogdan did and how Kotobotov and White GloveCoals kept on improving submission on submission.

As to my approach I realised early on that this was a competition of 2 parts; the optimization problem, working out how best to pack the bags given the distribution of gifts but also a leaderboard hill climb probing gift weights, to build known high weight bags.

That meant using every submission and in my mind trying to identify high weight gifts to fill bags quickly. So that lead me to probe coal;  3 submissions a day yielding the weight of 3 pieces of coal. 

I could rely on the Kaggle community and Kernels to solve the optimzation problem, and true to form Dominic Breukers kernel broke which with my known high weight bags of coal put me near the top of the leaderboard mid competition.

However a week later probing coal a bag per submission meant that I just could not keep up. That lead to the poll bust bags approach which I shared. 

However I only followed that approach for 3 submissions before I knew it would give limited benefit. At that point I implemented a variation of it. I selected 300 bags that contained 3 books and 3 trains (expected bust rate 4% ish) and deliberately increased their bust rate to 40% by swapping a train for a bike for all 300 bags. 

I then applied my bust bag detector. Bust bags were returned to their original state but altered bags would weigh 10lbs  more. I'd calculated that this could give me an additional 1000 points and that's pretty much what it did.

Simple and reasonably effective but clearly not the winning strategy..!
