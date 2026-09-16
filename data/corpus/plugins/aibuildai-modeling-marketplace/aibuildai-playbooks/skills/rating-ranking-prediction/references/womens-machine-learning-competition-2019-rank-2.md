# 2nd place solution: too risky (don't try at home!)

Competition: womens-machine-learning-competition-2019
Rank: #2
Source: https://www.kaggle.com/c/womens-machine-learning-competition-2019/discussion/88402#latest-512525

First of all, many thanks to @raddar for making his script from last year public (and for using R)!
He helped me -and I suppose most of us- a lot, but also raised the bar. And thanks to all participants for sharing so many insights!

Second, I know nothing about NCAA and especially the women's tournament. I am only familiar with Seton Hall (alma mater of Nikos Galis, the legend of Greek and European basketball) and Loyola (my wife happened to be an exchange professor in Loyola Chicago last March during their once in a lifetime feat, brought me back a Ramblers coffee mug!), so I am far far away from being an expert! Which proved a good thing... 

Having such a good script as a base meant that the machine learning part was already close to optimized. This meant that I would need to find an approach that would give some competitive advantage under certain conditions == increase the risk.

Combining practically Raddar's model with an ELO based approach of my own, I saw that there were big differences in terms of strength and that I could rank the teams in rather distinct groups:
1-2.  Baylor and Notre Dame
3. Oregon 
4-5. Mississippi and Connecticut
6. Louisville
7. Marquette
8- everyone else

As I said, I know nothing about the NCAA. I only read in the discussions that the top teams are much better than the rest. So my strategy to make the model more confident was to give the first four teams in my ranking a 0.95 in all their games (ordering by strength for games between the top four), keeping the binary option for the Baylor- Notre Dame game.

There were a few surprises in the first 2 rounds, so my submissions did goodish, in the bronze medal area. In Sweet-16 the heuristic worked great (as I suppose happened with everyone, since there were not any surprises), so I rise to the silver medal area. In Elite-8 I get very lucky: my heuristic gave 0.95 to Oregon(seed 2) vs Miss. St. (seed 1) and the model still saw a win of UConn (seed 2) vs Louisville (seed 1).

The all-or-nothing moment was the Notre Dame- UConn semifinal. Having no idea about NCAA, I didn't know about the [Connecticut–Notre Dame women's basketball rivalry](https://en.wikipedia.org/wiki/Connecticut%E2%80%93Notre_Dame_women%27s_basketball_rivalry)
Had I known about it, I wouldn't have allowed the model to bet 0.95 on a Notre Dame win! 

Needless to say, the same strategy for the men's tournament was a real disaster!
