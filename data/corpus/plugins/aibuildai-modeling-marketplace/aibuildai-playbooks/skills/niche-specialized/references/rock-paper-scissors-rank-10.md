# 10th Place - Layered Multi-window Counting

Competition: rock-paper-scissors
Rank: #10
Source: https://www.kaggle.com/c/rock-paper-scissors/discussion/221512

Thanks to everyone for a fun and stimulating competition, I learned a lot from everyone's discussions and their notebooks.  

 [Here is my public notebook for this competition.](https://www.kaggle.com/johntermaat/layered-multi-window-counting-agent)

My agents had a collection of strategies with 2 counter-strategies each, and used an n-gram tree to keep track of historical patterns in the match.   Rather than using complex agents to build a super-agent, I focused on extremely simple strategies, like basing a move on what I played last after an ngram + some spin.  For picking which strategy to use on each step, I discovered with time that making small, incremental changes to my agent rarely yielded as much payoff as trying lots of vastly different experiments, many of which went nowhere, but some of which did.  Since there are a lot of commonalities between various approaches to this competition, I want to focus on two concepts that I think worked well with my agents. 

1.  **Searching the record of a strategy with its own n-gram search.**  This is what the "FollowupTree" class does in my code.  For example, if a strategy got its last three predictions wrong, instead of just counting the losses, we look to see how well it's done in the past after getting 1, 2, and 3 predictions wrong.  Likewise with other patterns of wins and losses.

2.  **Multi-window counting.**  Since many agents might use some kind of "decay" or a rolling window, I attempt to find that window by examining a whole collection of rolling window counts. In the code, I refer to "weight schemes," which are collections of rolling windows that use a weighted average to return a score. Since we can think of a "strategy" as an action-selection and a record of hits and misses, we can use weight schemes to convert a collection of strategies into a new collection of strategies (the diagram below is intended to show how).

When we layer this technique upon itself, the agent seems to be better equipped to evade detection, and to dynamically target different unique opponent strategies. But it is also prone to more random fluctuation, which can lead to sporadically losing matches. 

[diagram]

**"Training"**

I created a "json_agent" that used historical episodes to play against my agents.  With multi-window counting, I would add a much larger collection of windows to the agent so that it was far too slow to participate in actual competition, and then count which windows had the best records in each match.  Then I used these windows in the actual match.

[screenshot]

**I learned a lot and look forward to learning more in the Hungry Geese competition.  Hope to see you there too!**
