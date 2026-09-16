# 3rd place solution - attack the opponent's second shipyard

Competition: kore-2022
Rank: #3
Source: https://www.kaggle.com/c/kore-2022/discussion/342296

Firstly, I'd like to thank @bovard and all participants. During the competition, I had great fun repeating the improvement of the agent.

# Summary
As well as many other participants, my agent is based on the solution provided by @egrehbbt and I improved the operations such as mining, building shipyard, defending shipyard, and so on. 

Overall, many features can be computed analytically, so there seemed to be few that would benefit from using statistical methods such as reinforcement learning. It took about three weeks to prepare the reinforcement learning, but the benefit seemed not to be very large, so I decided to change the direction after about a week of learning.

I think many operations are almost the same as other agents in the gold and silver medal. On the other hand, attacking shipyard operations would be relatively characteristic and this contributed to pushing my agent into the prize range, so I'll concentrate on the explanation of this operation and only briefly describe the others.

# Attack the Second Shipyard
The following is the agent obtaining the best score at the final evaluation. 

https://www.kaggle.com/competitions/kore-2022/leaderboard?dialog=episodes-submission-27412843

As you can see, in most matches in which the agent wins, the agent succeeded in attacking the opponent's second shipyard. This tendency holds even when the agent matches with agents in prize winners. On the other hand, the agent rarely loses matches due to the failure of the attack on the second shipyard. This would mean that judging whether the attack would be successful or not was very accurate and effective even in the high rating range. For this accurate judgment, some preparations were necessary. 

## Simulator
At the beginning of the competition, I implemented the simulator. This estimates the board status for the next 25 turns in the case that both players don't give any instructions. One of the characteristics of the competition is that it's easy to estimate the future state of the fleet with some accuracy since the flight plan of the departed fleets has already been determined. Therefore, the estimated result was very valuable. 

## Estimation
At every turn, the agent estimates how many ships are needed to attack each opponent shipyard and how many ships are available at each allied shipyard. 
### Estimate how many opponent ships can be used to defend
By the simulation result, how many ships at most can be in the target shipyard and its surrounding shipyards at each turn can be estimated. Therefore, the agent estimates how many ships the opponent agent can use for defending depending on the turn to start the attack. 

### Estimate how many opponent ships can be generated
In addition to the existing ships, generated ships before the attack is completed can be used for defense. Therefore, the agent calculates how many ships to use for defense can be generated based on the turn to attack, estimated kore at each turn, and each shipyard's ability to generate ships at each turn.

### Estimate how many allied ships can be used for attack
By the simulation result, the agent can estimate the number of ships to be used for the attack at each turn. In case the agent can utilize the collision with an allied fleet explained below, the agent includes the number of ships of the collided allied fleet.

## Collision with an allied fleet
To increase the number of ships for the attack, the agent searches the allied fleet in the middle of the pathway to the target opponent's shipyard to be attacked. This search is relatively easy if the agent has the simulation result because the agent can get the information of which turn and where allied fleets will exist. If there is a fleet in the middle of the pathway, the agent selects the path plan to collide with the allied fleet.

## Judgment of attack
Based on the estimated number of ships needed to attack and of ships available for attack, the agent decides whether it launches the fleet for attack or not. In addition, if there is a chance that more than the number of ships needed for the attack will be available within 8 turns, then the agent waits for the turn and doesn't launch the new fleet.

## Keep the number of ships at the shipyard
By the algorithms explained above, the agent can select the attack operation with precision if it happens to come across a situation where the attack is enabled. To increase the chance of the situation where the attack is possible, the mining plan was decided, so that the number of ships that the opponent's nearest shipyard would have would be secured after at least seven turns at each shipyard. This strategy degrades the mining efficiency somewhat, but increasing the chance of attack contributed to the win rate.

# Other Operations
Apart from attacking shipyards, the following operations were used:
- mining
- defense
- move
  
If there is a shipyard at risk of being taken if attacked, the agent moves some ships from a shipyard with a sufficient number of ships to the shipyard.
- convert
- attack fleet
- escape

If it is deemed likely that the shipyard would be taken, the shipyard launches a fleet to the safe shipyard.


=============

Again, I enjoyed the competition very much and was able to get my best placement! 

If you have any questions about my agent, please ask me freely in the comment. Thanks!
