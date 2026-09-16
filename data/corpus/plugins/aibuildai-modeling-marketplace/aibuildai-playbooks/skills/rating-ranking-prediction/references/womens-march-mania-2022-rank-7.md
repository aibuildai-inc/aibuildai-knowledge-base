# 7th Place Solution

Competition: womens-march-mania-2022
Rank: #7
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/318532

Similar to others, being aggressive with predictions is what brought me up the leaderboard. It backfired in the Men's competition (I placed 899th, thanks Kentucky and Iowa!), but it worked in my favor here. Ultimately to place at the top of these March Madness competitions, you need to introduce more volatility to your universe of potential outcomes - you might have a core model that can consistently place around the top 25-30%, but I don't believe you are going to find some magic math to consistently place in the top 10% (if you have, let's talk! :D)

I use an ELO ranking approach, very similar to 538's rankings. I shared a notebook back in the 2019 competition for parameter tuning the ranking - I use a version of this each year to think about how to set parameters for what I ultimately use in the competition. [https://www.kaggle.com/code/travisbuhrow/elo-scenarios-simulator-womens/notebook](https://www.kaggle.com/code/travisbuhrow/elo-scenarios-simulator-womens/notebook)

From there, I made the following "aggressive" manual adjustments:
- Increased UConn's rating 175 points - this moved them from my 4th rated team to my 1st rated team, but really the intended effect was to move them into the top "tier" of my rankings, so that they were considered mostly equal to Stanford and South Carolina. 
- Gave 1s, 2s, and 3s a 99.9999999% chance of winning round 1. LSU came very, very close to blowing this up! I'm also glad in hindsight that I didn't move on to doing this for round 2 games, because I would have ultimately given the 2s the auto-win... and Iowa/Baylor would have ruined the submissions.
- Took the Round 1 Nebraska/Gonzaga game, which my model had as a 50/50 game, and submitted one entry assuming Nebraska would win, and the other assuming Gonzaga would win.

Thanks to the admins for hosting another year of these March Madness competitions - some of my first exposure to data science and Python came with competing in the 2018 competition. It's a wonderful starter competition for newbie data scientists with interest in sports, as the volatility in outcomes means there's a little less importance placed on technical data science/coding skills, and a little more importance placed on domain knowledge and submission strategy. These competitions got me hooked on learning more about data science, and I'll be forever grateful for that!
