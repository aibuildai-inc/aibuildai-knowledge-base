# 22nd Place Solution - The RPS Gambit

Competition: rock-paper-scissors
Rank: #22
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221621

Whew! What a contest this has been right from the start. Competitive, energetic, nerve-wracking, and yes extremely informative. This is the first Kaggle competition that I participated in. Post the start of this, I did try my hand at the other open Kaggle competitions, but I got so caught up with RPS that I pushed all the others aside.

The past 3 months have been surreal, with each day leading to a quest for a better solution and a surge of excitement and tension as a solution is coded up. If you've watched the Netflix show, "The Queen's Gambit", you'd understand how I was for the last 3 months. I was so caught up by this competition, that I'd visualize strategies before sleeping, or I'd constantly be lost in thought, trying to think of improvements wherever I was. I couldn't stop myself from getting extremely caught up with this competition. 

The concept of the competition was pretty simple: make a bot that plays RPS and beats its opponents. I never realized that so much thought could go behind creating an RPS agent. There were various stages in the competition for me, each having a certain methodology I was working on. 

## Stage 1
This was during the first 2 weeks of the contest. I had no idea how to go about it, (remember once again, this is my first Kaggle contest), so I started reading the public notebooks being made available and the early discussions. After a while, I understood the functioning of the contest, and how to write a simple agent using the `obs` and `conf` parameters. But I then was quite intrigued by 3 public notebooks at the time, [Not So Markov](https://www.kaggle.com/alexandersamarin/not-so-markov), [MAB](https://www.kaggle.com/ilialar/multi-armed-bandit-vs-deterministic-agents), and [Memory Patterns](https://www.kaggle.com/yegorbiryukov/rock-paper-scissors-with-memory-patterns). I mainly played around with these notebooks at the time

## Stage 2
This is when I came across RPSContest, a little before everyone else started discussing it on the forums. By this time, many of y'all had started writing multi-strategy agents using the MAB as boilerplate, but I was still not clear about the functioning of a MAB, so it didn't strike me at the time. I wrote my own template to make these RPSContest bots compatible for the Kaggle contest, and then I started rigorously testing these. This is when I made use of the RPS: Agents Comparison notebook. During this period I shot up to the top 10 for about 2-3 weeks after tweaking multiple bots from RPSContest. The top-scoring ones then were [Random drop switch](http://www.rpscontest.com/entry/159004) and [Prognosis Modeling](http://www.rpscontest.com/entry/315003). Towards the end of this stage, I come across [this code](http://www.rpscontest.com/entry/547001) by [ben haley](http://www.rpscontest.com/authorSearch?name=ben+haley), who uses 2 strategies like an elementary ensemble. And this is a major turning point for me. I started building something similar except with 3-4 strategies. Basically keeps score of each strategy, plays the one with the highest score, and the scores of all strategies are reset to 0 after regular intervals

## Stage 3
Now we've reached mid-December. The competition is in full swing with about 800+ participants, and strong strategies are beginning to emerge. The Top Player Strategies thread really kicks off a phenomenal spectrum of strategies and approaches, and it is during this time I realize the importance and impact of a multi-strategy agent. And this was when I created my MAB-based multi-strategy selector. This is what I continued till the end of the contest. Throughout this stage, I was testing various strategies, the selectors and meta selectors, and I think by the end I had tried about 150+ various permutations and combinations of strategies. Like @returnofsputnik mentioned in his thread, I hardly submitted the same agent more than twice, rather I always tweaked parameters and then submitted the next version of the bot.


# My Top Approach:
So yes like many of y'all had implemented, I too have a pool of strategies (all modifications of RPSContest bots), and I have various methods of selecting the best agent from them. I have few things in common with multiple people's solutions, yet there are slight differences in how I added them into my selection mechanism.

I had 16 different strategies in my ensemble that took the opponent's last actions and returned the move to play next + 16 of the same strategies, except these take my last and then play the move that beats the output from these. So in total, I had 32 agents in my pool of strategies

### Strategies used and tweaked (credit to all authors)
- [Rosa Guadeloupe](http://www.rpscontest.com/entry/218004) 
- [Random drop switch](http://www.rpscontest.com/entry/159004) 
- [RPS_Meta_Fix](http://www.rpscontest.com/entry/5649874456412160)
- [dllu1_defensive](http://www.rpscontest.com/entry/503001)
- [meta-sgd-markov-full-ew](http://www.rpscontest.com/entry/5728994095792128)
- [Cyborg Duckling](http://www.rpscontest.com/entry/40001)
- [iocane-antidote](http://www.rpscontest.com/entry/146003)
- [rpsCrack](http://www.rpscontest.com/entry/334001)
- [zai_switch_markov1_ml](http://www.rpscontest.com/entry/540002)
- [db9](http://www.rpscontest.com/entry/315003)
- [RL_features](http://www.rpscontest.com/entry/5638703112257536)
- [wisard4_ai3](http://www.rpscontest.com/entry/871002)
- [bayes13_fix](http://www.rpscontest.com/entry/152030)
- [IocaneBayes2](http://www.rpscontest.com/entry/5127780110958592)
- [rmf](http://www.rpscontest.com/entry/5634661371871232)

And a special mention to @ilialar for the amazing [MAB notebook](https://www.kaggle.com/ilialar/multi-armed-bandit-vs-deterministic-agents) that served as a solid foundation to many contestants!


### Random moves?
Based on multiple discussions over the course of this competition, I did not use random generators to generate random output in my selector mechanism. Rather, I used sequences of numbers from prominent mathematical constants, as was done in the Pi bot. The constants I used were:
- [Pi](https://en.wikipedia.org/wiki/Pi)
- [Golden Ratio](https://en.wikipedia.org/wiki/Golden_ratio)
- [Natural Logarithm of 2](https://en.wikipedia.org/wiki/Natural_logarithm_of_2)
- [Euler-Mascheroni constant](https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant)
	
I had various sequences of size>1000 from these numbers. I used this in place of the `random.choice` option


### Selection mechanism

The first 90 moves are generated randomly (from the above sequences). I continuously monitored the scores of all my sequences and played the one with the current highest score. Not that it makes a difference as to which sequence is played, but for the sake of selection, I chose to do this
After this, my strategies now have enough data, to begin with. 

I have 6 performance lists, each scoring the strategies a different way. Performance3 uses *Dirichlet* samples from the `win, loss, tie` distribution of all strategies. 
Here are how the rest of the performances are evaluated
```
for i,c in enumerate(outputs):
            performance[i] = [{1:performance[i][0]+1, 0: max(performance[i][0]/2,2), -1: 2}[values[input+c]],  
                    performance[i][1]+values[input+c]]
            if values[input+c] == 1:
                performance2[i] = performance2[i] * 0.9 + 1
                performance4[i] = performance4[i]* 0.91 + 3
                performance5[i] = performance5[i]* 0.91 + 4
                performance6[i] = performance6[i]* 0.91 + 3
            elif values[input+c] == -1:
                performance2[i] = performance2[i] * 0.9 - 1
                performance4[i] = performance4[i]* 0.91 - 3
                if performance5[i]>10:
                    performance5[i] = performance5[i]-5
                else:
                    performance5[i] = performance5[i] - (performance5[i]*0.48)
                performance6[i] = 0
            else:
                performance2[i] = performance2[i] * 0.9 - 0.34
                performance4[i] = performance4[i]* 0.91 + 1
                performance5[i] = performance5[i] - 1.6
                performance6[i] = 1
        

        idx = performance.index(max(performance, key=lambda x: (x[0]**3)+x[1]))
        idx2 = performance2.index(max(performance2))
        idx4 = performance4.index(max(performance4))
        idx5 = performance5.index(max(performance5))
        idx6 = performance6.index(max(performance6))

```

The argmax was taken for the 5 performance lists. These outputs were passed to a final scorer function as follows:
```
answers = [outputs[idx],outputs[idx2],outputs[idx4],outputs[idx5],outputs[idx6]]
        for scorer in range(5):

            if values[input+scoring_state[scorer][1]]==-1:
                scoring_state[scorer][0] = scoring_state[scorer][0]*0.93 - 1.5
            elif values[input+scoring_state[scorer][1]]==1:
                scoring_state[scorer][0] = scoring_state[scorer][0]*0.93 + 1.2
            else:
                scoring_state[scorer][0] = scoring_state[scorer][0]*0.93 - 0.7
            scoring_state[scorer][1] = answers[scorer]

        best_sc = [scoring_state[i][0] for i in scoring_state]
        for k in [0,1,2,3,4]:
            #probe = scoring_state[k][0]/(scoring_state[k][0]+scoring_state[k][1])
            if scoring_state[k][0] == max(best_sc):
                best_a = k

```

I also added a switching component, both when winning and when losing. This is called `win_switch` and `lose_switch`. If my score is greater than -13, then I play the above method, but after every 30 moves, `win_switch` changes its value to 1, and I play 5 moves from the Dirichlet sampling. And then `win_switch` again is toggles to 0 and the process repeats

There is also another variable called `tog`, which stays at 0 as long as the score is greater than -13. This is used so as to ensure that the performance lists are updated only when a deterministic move is made, else the scoring will get skewed by the random moves I play. 
If the score is less than -13, tog takes either value 1 or 2. Here I use different techniques.
	- Play a random move from my sequences
	- Play a weighted random move, by choosing the strategy with the highest weight (These weights are updated every move)
	- Sample from *Dirichlet* (which is `performance3`)
	- Play my normal strategy

This image sums up the structure of my approach (apologies for image quality - do open it in a new tab)
[Nick John Strategy]


**My solution Notebook - [RPS Gambit - 22nd position Silver solution](https://www.kaggle.com/nikhiljohnk/rps-gambit-22nd-position-silver-solution)**

Throughout this competition, I have been oscillating between ranks 7-50 for the major part of the contest. But one thing I did observe is that I always had a bunch of my submissions keeping me afloat. Even now at the end of the contest, I had 28 agents in the medal zone, with 21 bronze and 7 silver.

But I think the highlight for me was learning from all you guys, and appreciating how Stats has such a presence even in a game like RPS. I've been motivated to read up more about stats and Game theory, and I hope to get familiar with RL soon so that I can join the Hungry Geese contest (for now I'm completely clueless about what to do in that contest)

I want to specially mention @tonyrobinson, @taahakhan, @johntermaat, @glazed, @chankhavu, @jbomitchell, @superant, @kdanielive, @stassl, @georgstreich, @booooooow, @returnofsputnik, @neibyr, @jamesmcguigan, @robga, @purplepuppy, @piotrekga, and @karham for all your contributions to this competition through the extremely insightful discussions and notebooks. I enjoyed reading and expanding upon the competition boundaries with you all. Hoping to see y'all again in future competitions. And yes thank you @sam and the entire Kaggle team behind this competition.
