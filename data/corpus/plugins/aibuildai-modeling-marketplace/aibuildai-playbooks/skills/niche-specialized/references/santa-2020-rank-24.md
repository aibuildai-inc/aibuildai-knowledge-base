# My Approach - non-parametric probability estimate + copy

Competition: santa-2020
Rank: #24
Source: https://www.kaggle.com/c/santa-2020/discussion/216779

Before going into any details. I wanted to say the game turned out to be more interesting than I imagined (full of greedy/copy/deception though...). I want to thank Kaggle for hosting this competition and thank all people who shared their notebooks and thoughts. It was funny to see all the name changes by nagiss. Big fan here.

In terms of my approach, I think there are several key points:
- A non-parametric probability estimate using replay data
- Some simple copy/exploit rules
- Luck... of course, all you need is luck :)

The most important thing is probably a non-parametric probability estimate. I basically run a group-by mean/std over replay data to estimate the average bandit probability. The dimensions include the following:
- a0 = total number of pulls by myself
- a1 = total number of pulls by the opponent
- r0 = total number of rewards for the bandit
- p = bandit threshold for the bandit

All these values can be read from replay files for both agents and each time step and each bandit. Then I have this big data frame for each replay and drop duplicates and do groupby:
- df.groupby(["a0", "a1", "r0"])["p"].mean()
- df.groupby(["a0", "a1", "r0"])["p"].std()

On top of this, I then average the estimates across all the replays to get a stable estimate. I tend to think of this approach as doing Monte Carlo simulation for all those possible scenarios of (a0, a1, r0) without having to specify a closed-form formula. This simple method turns out to be pretty effective and accurate (though I didn't measure precisely how well against other parametric or tree-based fitting methods).

I also tried to add some other dimensions in the group-by:
- Whether the bandit is just pulled by the opponent (0/1)
- Whether the bandit is just pulled twice continuously by the opponent (0/1)
- Number of continuous pulls by the opponent on this bandit
However, I didn't see a significant improvement in terms of performance.

Looking back I can see some gold-zone players probably tried to deceive other people by performing certain actions, which would mislead my model to perceive the probabilities in a wrong way. Also, this estimate would be sensitive to the data you feed in.

In terms of copy rules, my best agent so far uses a rather brute-force copy:
- If t < 200, just copy everything
- If the probability of the bandit is higher than the median of the probability vector for all bandits, just copy.
- If the opponent pulled the same bandit twice and I haven't pulled the bandit a lot, just copy. By a lot here I mean 20/30+, without this protection, the two agents might stupidly copy each others' move endlessly (I've seen it, I promise...)

I think copy can be a very effective technique here, in the sense that you essentially hide all the information, but you are at some level of a disadvantage given the 0.97 decay after each pull. But with good enough skill in exploiting, you can catch up and make a good score in the end. 

Also, I would like to share some of the ideas that I thought about/tried a little but didn't have enough time to fully explore (some already covered by top solutions):
- LSTM with opponent action + self-action + self-reward
- dynamically recognize the opponent type and try to attack its weaknesses accordingly
- RL with maybe some DQN agents trained from self-play
- LGB with certain features
- Dynamic exploit/explore adjustment
- Try to deceive/mislead the opponent

Finally, I think the game overall has a very interesting setup. An important part of the game is watching how the agents evolved as the game goes on. The monkeys learned how to be greedy/copy/deceive etc. It's cruel to see your previously best agent now can't even make it to 1000. How alpha decays and how luck dominates!

Thanks for reading. Wish everyone gets more candies.
