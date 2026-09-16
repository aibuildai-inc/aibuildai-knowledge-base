# 10th place solution (Team Breakfast)

Competition: kore-2022
Rank: #10
Source: https://www.kaggle.com/c/kore-2022/discussion/340159

This a summary of the 10th place solution by Team Breakfast, and of our journey to get there. Our agent is purely rule-based.

# Preface

@vladikrajnak and I started this competition for reinforcement learning. Since the action and observation spaces are huge, our initial goal was to have an RL-based macro strategy for long-term planning, while using heuristic rules for micro-management. We started with @egrehbbt solution and tried to improve on it. Our rule-based agent started to perform quite well, even being in the top 5 for a while, so we forgot about RL and focused on a pure rule-based agent.

On a side note we also considered using tree search, but the Kaggle simulator was way too slow for that, and we had neither the interest nor the skills to rewrite a much faster one.

Like many others, we are particularly grateful to:
- @egrehbbt for sharing their 1st place solution for the beta competition. It was a great starting point in terms of strategy.
- @jmerle for providing an amazing visualization tool. It helped us tremendously with debugging and analyzing games.

Our top agent is one that we submitted 2 weeks before the end of the competition. We implemented many changes in the last 2 weeks: delayed first expansion with an aggressive early attack, safer routes (to prevent fleet attacks), safer launches (to prevent shipyard attacks after launching many ships away), setting the miners destination so as to dispatch them where needed (either shipyard with more kore mining potential or weak shipyard). Yet none of them had a significant effect: to our surprise all our latest agents ended up with similar scores (1550-1600), despite behaving quite differently. 

Compared to the top players' agents, we think we lacked the coordinated attacks, and shipyard abandonment when about to be taken over. These were presumably key features for reaching a higher "layer" in the leaderboard. We actually had those on our to-do list, but we did not have time to think them through carefully, and chose to prioritize other features in the limited time we had left.

# Building our agent

Our first focus was to make all @egrehbbt's estimates (kore-equivalent value of a mining route, number of ships for a successful attack of shipyard, etc) more accurate.

We used board.next() to predict the future board states (updated after every launch, provided we had enough computing time left in the current turn), and we implemented an accurate (though very time consuming) route evaluation algorithm. This slowed down the code a lot, but improved our mining, which we thought was essential (especially to be able to dominate very early in the game).

On the other hand, our estimates of the number of ships needed for a successful shipyard capture, or the max number of ships we can send mining without endangering our shipyard, did not work as well as we hoped. Probably because they were too conservative (based on a worst-case scenario) (as also pointed out by [qihuaz](https://www.kaggle.com/competitions/kore-2022/discussion/340157)).

We kept the sequential structure of @egrehbbt's agent, but reorganized it a bit, and rewrote most of the functions to make them more efficient, accurate, and/or to allow some anticipation.

### 1. Defend shipyards
Similar to the original: spawn if under attack, and if needed launch help from neighbor shipyards. For launching help, we tried various strategies: 
- launch all available at the last moment (original)
- launch all available right now
- launch only if the shipyard under attack cannot self-protect by spawning alone in the next turns
- spawn/gather incoming miners until there are enough ships to successfully defend, then launch the number needed

While short-sighted, the second method turned out to perform best. Maybe because it allows moving ships around and mining on the way as side effects.

Improvement needed: flee if we cannot defend, instead of spawning. 

### 2. Capture shipyards
In addition to improving the estimate of the number of ships needed, we also added anticipation: if an attack is possible within in a small number of turns, we stop mining and spawn in anticipation of the future attack launch.

Improvement needed: coordinated attacks.

### 3. Direct attacks / Adjacent attacks
The goal of direct attack is to steal kore from an opponent's fleet, while adjacent attacks are suicide missions aimed at destroying ships through double damage. We always prioritized direct over adjacent attacks, and used relevant (different) scoring for each. For adjacent attacks, we also choose a route that minimizes the kore gathered along the way.

One question here is how much risk to take. Attacking a fleet next to an enemy shipyard is certainly going to result in a counter-attack. On the other hand, launching fleets only if they are 100 % safe results in essentially no attack being launched.

However optimizing this balance does not seem crucial, as both our careful and our careless agents essentially scored the same. We would actually argue that fleet attacks are not essential to wining (unless maybe if the opponent is completely careless/very agressive).

###  4. Expansion
We tried to find a good expansion criteria and new shipyard location, also trying to adapt to the opponent's behavior (e.g. attacking agressively if they are first to expand, expanding more conservatively if we are first).
But in the end our best agent expands using very basic rules:
- expand when possible (enough ships in shipyard)
- choose the best location within a distance 4-8 of the current shipyard based on the most kore nearby, forbidding locations which are closer to an opponent's shipyard than to ours.

We also added some anticipation there (if it's time to expand but there are not enough ships in the shipyard, stop launching miners and spawn instead, until we can launch).

### 5. Spawn/Mining
We kept @egrehbbt's criteria for choosing between max spawning and mining, and also chose the best route based on expected kore per step.

We thought having a more accurate expected kore would give us an advantage, but looking at other published agents it does not seem as important as we thought. For the number of ships to launch, we launch min_fleet_size if the route is considered safe (crude heuristic, though not the original one), and all available ships otherwise.

We added the option to prevent a launch that would endanger a shipyard (a young shipyard containing zero or few ships could be easily captured), it is unclear whether it is better with or without.

We spent a lot of time in the last 2 weeks building a more extensive list of routes to choose from, making the evaluation routine faster, optimizing the choice of the fleet size, choosing destination shipyards in a smarter way... but it did not make the slightest difference in the scores.


# Conclusion

We did not learn about RL as planned, but we learned a lot about debugging and coding in Python. In particular the 3-second time constraint forced us to identify bottlenecks and optimize calculations (we stuck to Python though), and to make strategic choices about where to allocate that time.
  
We had a lot a fun and we hope there will be a season 2!
