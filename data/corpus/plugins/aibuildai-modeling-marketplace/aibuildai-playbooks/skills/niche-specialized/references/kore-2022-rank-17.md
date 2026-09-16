# Rule based solution (15~20th Place)

Competition: kore-2022
Rank: #17
Source: https://www.kaggle.com/c/kore-2022/discussion/336826

# Introduction
At the beginning, I would like to thank Kaggle teams for serving the great competition. I learned a lot of techniques and knowledge from this competition. My teammate @seikinakamura gave me some advice and encouraged me.

# Rule Based Solution
My solution is based on the excellent Kore Beta 1st place solution (https://www.kaggle.com/competitions/kore-2022-beta/discussion/317737) and I added various rules from my idea. I didn’t use RL techniques because of many available actions.
My rule based solution is described below.

- expected future field
First, I calculated the board after 20 turns using board.next(). This enabled me to expect the fleet position, the number of shipyards, and so on.

- defense
Mainly based on the Kore Beta 1st place solution. I took into account the number of ships the closest opponent shipyard has. We can specify the fleet which is going to be converted into a shipyard, so we can send additional fleets to the converted shipyard in the future.

- fleet attack
I launched a fleet to destroy an opponent fleet to gain the kore. If the distance between the target and my shipyard is closer than that of the enemy, the target can be destroyed without fail.

- shipyard attack
If my shipyard power is stronger than that of the enemy, we can occupy the opponent shipyard. The opponent shipyard power is the sum of these three powers: the ship count the opponent has, max ships to spawn, and sended ships from other shipyards by the time the launched fleet reaches.

- expansion
This was also based on the Kore Beta solution. The new shipyard is not converted around existing shipyards due to the convert cost.

- greedy spawn
If the shipyard can spawn maximum ships, then spawn ships. However, if I don’t plan to create new shipyards, the ships can be spawned up to 21 (This is equal to 7 length of a flight plan).

- sending ships to friendly shipyards
The shipyard surrounded by allied shipyards is safe. The ships belonging to the shipyard are sent to other shipyards in case of attack or mining.

- mining
The launched fleet mines as many kores as possible per turn. It’s very important for me to decide the fleet size (the number of ships). I calculated the fleet size and the mining turn in terms of the distance between my shipyard and the opponent shipyard.

- spawn
If the shipyard action is not decided, the ships are spawned.

# Code
I shared my solution below.
https://github.com/mhal-teddy/kore2022_public
