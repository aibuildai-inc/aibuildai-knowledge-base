# 1st Place Solution - Where is my bag's Journey

Competition: rock-paper-scissors
Rank: #1
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221502

**EDIT**:
- I rephrased the description of agent group C to avoid some misleading thanks to @stassl

First of all, it is a fun and interesting competition as well as learning experience for me, I spent lots of time working on this problem and I really enjoyed it!

Along with the solution, I would like to share how I approached this problem from day 1 and hope my journey can let you know that it might be not that hard to achieve a high rank in a competition.

##TL;DR
multi-armed bandit with list of [rpscontest](https://web.archive.org/web/20200521083624/http://www.rpscontest.com/leaderboard) agents + 1-step & 2-steps ahead transformed agents + dirichlet distribution.

## Solution
Here is the my brief solution for my top 5 submissions (or see from this [link](https://bow-public.s3.us-east-2.amazonaws.com/image/RPS/solution.jpg)) :


Here is the list of agents I used for my top submission:

| Agent                                                                                              | Regular | 1-Step Ahead | 2-Steps Ahead |
|----------------------------------------------------------------------------------------------------|---------|--------------|---------------|
| [DTC](https://www.kaggle.com/alexandersamarin/decision-tree-classifier?scriptVersionId=46415861)   |         |       o      |               |
| [rfind](https://www.kaggle.com/riccardosanson/rps-simple-rfind-agent)                              |    o    |       o      |       o       |
| [#01](https://web.archive.org/web/20200812062252/http://www.rpscontest.com/entry/885001)           |         |       o      |       o       |
| [#02](https://web.archive.org/web/20200812060710/http://www.rpscontest.com/entry/498002)           |    o    |       o      |       o       |
| [#03](https://web.archive.org/web/20200220023240/http://www.rpscontest.com/entry/5649874456412160) |    o    |       o      |       o       |
| [#04](https://web.archive.org/web/20200812073700/http://www.rpscontest.com/entry/5728994095792128) |    o    |       o      |       o       |
| [#05](https://web.archive.org/web/20200521083624/http://www.rpscontest.com/entry/892001)           |    o    |       o      |       o       |
| [#06](https://web.archive.org/web/20180910094734/http://www.rpscontest.com/entry/5689531432960000) |         |       o      |       o       |
| [#10](https://web.archive.org/web/20201021153705/http://rpscontest.com/entry/342001)               |         |       o      |       o       |
| [#11](https://web.archive.org/web/20200521083624/http://www.rpscontest.com/entry/5716517316460544) |         |       o      |       o       |
| [#24](https://web.archive.org/web/20201022114135/http://rpscontest.com/entry/202002)               |         |       o      |       o       |
| [#27](https://web.archive.org/web/20201021155550/http://rpscontest.com/entry/315005)               |    o    |       o      |       o       |
| [#33](https://web.archive.org/web/20201021160032/http://rpscontest.com/entry/555004)               |    o    |       o      |       o       |
| [#36](https://web.archive.org/web/20200521083624/http://www.rpscontest.com/entry/887001)           |    o    |       o      |       o       |
| [#49](https://web.archive.org/web/20201022115133/http://rpscontest.com/entry/175002)               |         |       o      |       o       |


## Local Evaluation
Unfortunately I was unable to find a consistent way to evaluate my local agent because LB is noisy. But I still shared it anyway. Here is the list of benchmark I used:
- [Decision Tree Classifier](https://www.kaggle.com/alexandersamarin/decision-tree-classifier?scriptVersionId=46415861) by @alexandersamarin
- [Random Forest Classifier](https://www.kaggle.com/jumaru/random-forest-random-rock-paper-scissors) by @jumaru
- [RPS Contest #4](https://web.archive.org/web/20200812073700/http://www.rpscontest.com/entry/5728994095792128)
- [Geometry](https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic) @ superant
- One previous well-performed submission
- Two episodes' logs that beat me in the past

I used an ec2 instance with 8+ CPUs and ran 20 games for each benchmark. (~ 3000 games per day)

## Where is my bag's Journey

##### Phase I
(screenshot credited to @demche)

I wanted to learn some reinforce learning knowledge so I joined this simulation competition on Nov 6, 2020. 

I don't really like the idea of random (or Nash equilibrium) agent so the first agent I wrote was very similar to [(Not so) Markov](https://www.kaggle.com/alexandersamarin/not-so-markov). 

I had lots of "brilliant ideas" of RPS e.g. reactionary, counter-reactionary but none of them performed well on LB so I was a little depressed, then I looked into other's public notebooks.

##### Phase II

This [Multi-Armed Bandit](https://www.kaggle.com/ilialar/multi-armed-bandit-vs-deterministic-agents) notebook by @ilialar opened a new world to me and I believe the idea of MAB fits this competition so I started to code my own MAB structure. I took one entire day to code and another day to debug but eventually I made it and used the structure till the end.

I spent a lot of time tuning decay rate and reward of MAB but the score didn't change too much so I looked into the public notebook again and I noticed this [Decision Tree Classifier](https://www.kaggle.com/alexandersamarin/decision-tree-classifier?scriptVersionId=46415861) performed really well on LB at that time, so I begun to write 1-step-ahead (or beater) agent of it and added it to my MAB. Thanks to the beater agent I was able to reach Top 5%. 

##### Phase III

One day I came cross this [discussion](https://www.kaggle.com/c/rock-paper-scissors/discussion/197947#1085310) and realized oh there was [a RPS contest](https://web.archive.org/web/20200521083624/http://www.rpscontest.com/leaderboard) before! So I looked into it and thanks to @purplepuppy 's [notebook](https://www.kaggle.com/purplepuppy/running-rpscontest-bots) I was able to add contest agents to my MAB. My LB score was again improved!

##### Phase IV

I gradually have two core ideas in my mind:
**1.  To achieve a higher rank, the agent should beat as much public agents as possible.**
**2. A good agent cannot easily beat itself.**

Based on the first idea, I added beater(1-step-ahead) agents for RPS contest agents to my MAB; based on the second idea, I added 2-steps-ahead agents. 

So, my MAB was like: the 1-step-ahead agent beats my regular agent; the 2-steps-ahead agent beats my 1-step ahead agent; the regular agent beats my 2-steps-ahead agent.

Sounds really good, right? But my enhanced agent didn't perform well. After going through my MAB structure again and again, I realized the input action for each agent was wrong. Once I fixed it, my LB score was finally improved!

##### Phase V

At that time, my rule of picking agent was many if-else statements and I spent lots of time trying different strategies. One day I decided to delete all of the if-else statements and just keep beta distribution for agent selection. Surprisingly, it dominated over all of my previous agents!!  And it stayed at my highest score submission for a very long time.

While I was trying different combination of agents, @georgstreich mentioned [dirichlet distribution](https://www.kaggle.com/c/rock-paper-scissors/discussion/208242#1135937) which makes much more sense to me than beta distribution so I changed all of my agents from beta distribution to dirichlet distribution. This is how my current top submission is from.

###### Phase VI

This phase is mainly focusing on the idea of stacking two MABs together, one for agent selection and one for strategy selection, they were doing really well until [Geometry](https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic) agent came out :/ Unfortunately I was unable to frame a well-structured MAB to beat geometry agent so I was very anxious during the final two weeks. But I am glad to see some of my "veterans" still hang in there.

###### My Daily Life of RPS
Once I woke up, the first thing I did was to check RPS leaderboard and my submissions(I wonder how many of you were the same lol). Then I uploaded my submission candidates created last night to my ec2 instance to evaluate locally. In the later afternoon the evaluation was done so I checked the performance of each candidate and submitted top 1 and 2 candidates to the competition. If any of them was losing 2 games then I would resubmit the same agent. Once the deadline was passed, I started to create new agents for tomorrow...

## Acknowledgement
- @addisonhoward and @sam - Thank you for hosting this competition!
- @tonyrobinson - You are the pioneer of deep learning! And I always enjoy reading your postings.
- @purplepuppy - Thanks for bringing really great agents to the old RPS contest!
- @chankhavu, @returnofsputnik - so much fun to play the RPS with you guys haha!!!
- @stassl, @taahakhan, @glazed, @nikhiljohnk, @superant, @georgstreich, @robga, @jamesmcguiga - thank you guys for very helpful RPS discussion!!!

Thank you for reading here and please post any questions you might have. I'll see you in the next competition :D 

-Bow😄
