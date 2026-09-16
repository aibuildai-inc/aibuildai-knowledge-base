# 4th place solution - rule based

Competition: kore-2022
Rank: #4
Source: https://www.kaggle.com/c/kore-2022/discussion/340157

# The Journey

First of all, I'd like to thank the Kaggle team for hosting such an interesting competition. Given that most of the competitions with prizes on Kaggle are centred on supervised learning, e.g. computer vision, NLP, etc,  Kore 2022 is really a fun one in the sense that competing is just like playing a game and it is hard to get bored with it.

Like many others, I started off with Reinforcement Learning but soon switched to rule-based agents after discovering the Beta 1st place solution by @egrehbbt. I was quickly convinced that RL alone cannot win this competition. Even if someone managed to use RL to get to the top, it will probably need to be coupled with actions suggested by rule-based heuristics.

At this point, It is probably not a surprise that my agent is based on the Kore Beta 1st place solution by @egrehbbt, so huge thanks to @egrehbbt. It laid a super solid foundation for my agent. Even though I changed/added a lot of logic and rewrote most of the functions, that framework is just crucial and I am just standing on the shoulders of a giant.

Also a big thanks to @jmerle for his great visualization tool. I probably used it a thousand times. It made debugging so much less painful.

 
 
# The Agent

## Strategies

Before diving into the details, I would like to share some thoughts at the strategic levels:

- Defense is more effective than offence, and so is given higher priority
- Collecting kore that you cannot spend quick enough is futile. 
- Reactive actions (rather than proactive actions) can give you competitive advantages.


## Tasks

My agents try to finish the following tasks (almost) sequentially.

- defend_shipyards
- capture_shipyards
- expand

- defend_fleets
- suicide_attack
- roundtrip_attack

- greedy_spawn
- mine
- spawn


### Defend Shipyards & Capture Shipyards

Nothing fancy here. When a shipyard is attacked, I evaluate nearby shipyards to see if reinforcements from nearby shipyards are enough to defend the attacked shipyard. If so, mobilize them for defence.

When I want to attack/capture an enemy shipyard, I assume all its nearby shipyards would help with defence to their maximum capacity and I would only attack if I can win in that case. I also launch fleets from multiple shipyards to capture a single shipyard. I am aware that this is prone to adjacent attacks, but didn't have the time to fix it like @harmbuisman does in [his bot](https://www.kaggle.com/competitions/kore-2022/discussion/340035#:~:text=but%20then%20I%20managed%20to%20route%20the%20fleets%20to%20minimize%20adjacent%20attack%20risk).

one thing I never quite figure out is my own future spawns. When kore is limited and shipyards compete for the resource for spawning, I may miscalculate how many ships a shipyard can spawn and mess up my own defence.

### Expand

One basically needs to decide: 1) when and 2) where to expand.

**when to expand**: One should expand when kore collection is faster than the speed of consumption

**when NOT to expand**: If the ships/shipyards ratio would be significantly smaller than that of the opponent.

**where**: it is a balanced selection based on potential profit (nearby kores) and risks (nearby shipyards). 
- Only consider the cells (potential new shipyard location) whose two closest shipyards are friendly shipyards (or one if I have only one shipyard)

The candidate locations are then scored by the sum of nearby kores.
- 'nearby' is defined to be within a maximum radius, which is in turn determined by the number of total shipyards (more shipyards, smaller radius). 
- nearby kores are discounted if there is already a friendly shipyard near it.


### Defend fleets

If a fleet is under attack, check if the following is possible:
- launch a new fleet to join (merge with fewer ships) it to over-power the attacker
- launch a new fleet to absorb (merge with more ships) the attacked fleet so the attacked fleet changes path to avoid the attack
- launch a fleet for a suicide mission to weaken the attacker, so the attacked allied fleet can win the final battle.


###  Suicide attack & Roundtrip attack

suicide_attacks are the fleet launches that do not expect the fleet to come back to bases, and roundtrip_attacks otherwise.

**suicide_attack**
suicide_attack includes the 'adjacent attacks' that many of you are familiar with, but it also includes some other situations. For exmaple, send out a suicide_attack fleet to offset an incoming enemy "adjacent attacker".

Routes are evaluated to choose the one that collects the least kore.

**roundtrip_attack**

The roundtrip_attacks are enhanced with a few features:

- If the available ships are not enough for an attack, **actively check if afriendly fleet nearby can be absorbed to make the attack possible**.
- If the available ships are not enough for an attack, check if spawning more for a few time steps would make an attack possible
- If the same fleet can be attacked at a later time, do it later (more kore looted, less time for opponent to react).
- **the attack is implemented as a mining mission** that has to pass a specific cell at a specific time (or two if it has to absorb a friendly fleet before attacks) so that maximum kore can be collected along an attack mission.


### Mining

This part takes most of my time, in terms of both coding time and in-match computation time. Maybe too much time on it, but this definitely give me an advantage.

**Irregular mining routes**

Enhancing mining is one of the few things that I did at the very beginning since the advantage is obvious. I started exploring irregular mining paths (paths that are more complicated than rectangular or L-shaped) early and that allowed me to stay on the top of the LB for a while before other competitors caught up.

I precompute a set of mining routes (about 80k+, far less than all possibilities) for a centred shipyard, load it and adapt to the actual shipyard locations in run time. The way I evaluate the expected collected kore can be found in this [discussion](https://www.kaggle.com/code/solverworld/computing-speed-for-kore-harvesting-routes/comments). @solverworld did an amazing job  [profiling different approaches](https://www.kaggle.com/code/solverworld/computing-speed-for-kore-harvesting-routes/comments) .

After some optimizations (caching, precomputing), I still only manage to evaluate about 30k routes (all shipyards combined) within the 3s per step limit, so I had to prune the candidate routes using various heuristics.

**Risk control**

The other thing I did to enhance mining is to avoid routes that are not safe from enemy attacks. My way to achieve this is to maintain a 3-dim "net power map", indicating at each time step for each cell, how much damage can/would be made (positive for friendly power, negative for enemy)

A slice of it looks like this:



This map helps my fleets avoid most attacks, especially in the early game. However, it is not meant to be 100% safe. I found that if I try to avoid all "attackable routes", the routes would be too conservative and give away too many good mining opportunities.

I guess many top players have similar tools to manage these risk.


**Prioritize mining kores in "hostile areas"**

Soon after I started irregular mining routes, I found that is a double-edged sword. My mining is so efficient that I can easily deplete kores near my shipyards, while there are plenty of kores in "hostile areas" where mining is risky. While the safe and efficient mining gives me advantages in the early game, if I cannot finish the game early the depleted kore became a disadvantage. 

To solve this, I adjust the values of the expected kore mined. If the cell is located close to enemy shipyards, it is given a higher adjustment factor. The rationale is this: although it is riskier to mine in those areas, those are actually the kores that should be fought for. Unlike the kores near my shipyards, one more unit of kore mined by me in the risky area is one less unit of kore minable by the enemy.

=========

There are many many more rules and details, and it would be too cumbersome to explain them all, so I just picked some of the important ones and share with you in above. I am more than happy to discuss in more details if you have specific questions.

Thanks for reading and I look forward to see you guys in the next competition.
