# 20th place solution: rule-based economic model

Competition: kore-2022
Rank: #20
Source: https://www.kaggle.com/c/kore-2022/discussion/339972

Congratulations to all competitors. Here's a summary of my solution, which is entirely rules-based. I attempted to treat this as an economic modeling problem combined with path optimization (described at https://www.kaggle.com/competitions/kore-2022/discussion/336804)

**Momentum**
- Build a look-ahead "momentum" model of the board for the next 30 turns
- Build balance-of-power matrices which show maximum force projection for the players for each square over the 30 turn horizon

**Valuation**
- Estimate the net present value (NPV) of a new ship, assuming no combat, using a 1% discount rate
- Estimate the NPV of a new base, assuming the new ship NPV, existing production capacity, estimated income, and kore
- Zero out the new base NPV if I already have more shipyards than the enemy but also fewer ships, to avoid over-extension 

**Priority Spawning**
- Create a priority queue for new ship construction
- Any shipyard which is under active capture threat gets critical priority for new production as well as kore set aside for production in future turns, if needed
- Neighboring shipyards which could reach the threatened yard get high priority for new production
- Assign normal priority to all other shipyards
- Create SPAWN actions for high and critical priority shipyards only. If we run out of kore, add those yards back to the launch queue.
- If there are any critical or high priority spawns going on, disable new base building

**Launch vs. Normal Priority Spawning**
- In the early game, wait for enemy to make the first launch so we might be able to opportunistically get a bigger payoff from a foray into the mid-board
- If I have less than 21 ships at a shipyard and kore to spend, prioritize spawning over launching
- Otherwise, launch!

**Risk Adjustment**
- If I have a numerical advantage, adjust the balance-of-power matrices in my favor so I begin taking more risks (i.e. penetrating deeper into enemy territory than I would normally deem safe).

**Strategic Labelling of Home Bases**
- Estimate the economic value of each base (for me and the competitor).
- If the base is actively threatened, assign this value to the square for each time step prior to its capture
- Also assign the same value to destroying the inbound fleet
- If my current fleet is large enough to prevent the capture, ignore negative balance-of-power when defending home bases

**Strategic Labelling of Enemy Bases**
- Estimate the economic value of each base (for me and the competitor).
- Assign the value to this square but use a 1% time discount to encourage my fleets to capture it sooner

**Balance of Power Strategy Labelling**
- Note: this portion of the algorithm implemented in Cython for speed
- For each shipyard, including enemy shipyards, look at each arrival time step and its implications on the balance-of-power matrix for neighboring shipyards.
- If a home shipyard's balance-of-power is always positive, ignore it. It should be safe.
- If not, see if our arrival at a given destination in a given time step can shift the balance-of-power in our favor
- If it can shift it entirely positve, assign 50% of the yard's value to the destination
- If it can shift only some time steps positive, assign at most 5% of the base's value to the destination depending on the proportion of time steps which have flipped positive.
- Do the same for the enemy bases, however only assign the maximum value of the base which I am threatening since I can't simultaneously capture multiple bases with the same fleet.

**Create Trial Paths**
- Create a list of possible fleet sizes: full size, half full, 21, 13, 8, 5, 3, and 2. If I already have a very large number of ships at the shipyard, skip some of the smaller sizes.
- Create a "zombie" fleet size of "full size". Zombie fleets try to avoid mining kore and hone in on enemy bases or fleets as quickly as possible.
- Calculate minimum efficient time to be "ships at shipyard" divided by "fleet size"
- Calculate minimum bottleneck time as the turn at which our kore reserves will run out if we produce every turn. I.e. try to avoid arriving back home in a turn when the base would like to be producing new ships.
- Run optimal path finding algorithm on each possible fleet size, constrained by the minimum efficient time and minimum bottleneck time
- If our new base NPV is positive but we don't have enough ships, create a maximum time constraint so that fleets will coalesce into larger sizes so we can build the base.
- There is also effectively a maximum time constraint applied in the strategy labelling step, depending on which shipyard destination the fleet is heading to.
- Select the best path among the trial options, optimizing for best kore-per-time-per-ship among the paths with maximal strategic value, constrained by the maximum and minimums described above. 

**Zombie conversion**
- If my best path is an attempted base capture but there's some possibility it will fail, re-run the algorithm using the same fleet size as a zombie and use this path instead.

**Zombie march**
- Look for vulnerable enemy bases that my fleet ignored but which I could capture given enough air support from other fleets. 
- Do another trial path using a zombie fleet targeted at this base only. If this gives me a higher value, use this path instead.

**Base building**
- If my new base NPV is positive and greater than my best flight plan value, create a new base instead of my original plan
- Look for suitable base locations, ideally 6 squares away near lots of kore (by absolute value) and kore "seeds" of any size (i.e. future mining potential). Give higher weight to kore located on the same X or Y axis. Try to build dominance in the best areas (e.g. center, sides, corners) before expanding elsewhere.
- Use the path optimization algorithm to define the path to the new base location

**Normal priority spawning**
- For any shipyard not launching fleets, create a SPAWN action if we still have kore left
