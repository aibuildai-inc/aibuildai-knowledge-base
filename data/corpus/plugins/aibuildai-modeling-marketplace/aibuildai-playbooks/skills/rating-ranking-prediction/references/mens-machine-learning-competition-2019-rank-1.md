# 1st place solution

Competition: mens-machine-learning-competition-2019
Rank: #1
Source: https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/89150#latest-516634

First, I would like to express my gratitude to the organizers of this competition, and most importantly to raddar for sharing his winning solution in last year’s women’s competition, as it was extremely helpful to me. 

In this challenge, luck has had a very big role in determining the winning teams. In my approach to solving this, I decided to rely on raddar’s model for the men’s competition with some slight modifications to first make it more conservative, and only then take chances with a few first round games.

Originally, raddar’s model had 1-4 seeded teams beating 13-16 seeded teams with 100%. As this strategy rarely works in the men’s competition, I decided not to employ it. This turned out to be a good idea because of the UC Irvine upset.

Assuming this model would be enough for a decent position, but not enough to land in the top 5, I decided to try my luck by overriding 3 first-round games (assigning 100% probability to one team), as I felt that was a good trade-off between risk-taking and the possibility of improving the score. That’s where I indeed got lucky. Actually, this same approach did not work out too well for me in the women’s competition where I ended up at the 434th position out of 500 :D. 

Lastly, since I had the chance to compete with two submissions, I picked one 50-50 first round game to play both outcomes (as this could improve any model by ln2/63 = 0.011 points). 

Here is the link to my solution: https://github.com/salmatfq/KaggleMarchMadnessFirstPlace
I hope everyone enjoyed the challenge, for me, it was definitely exciting to watch it unfold.
