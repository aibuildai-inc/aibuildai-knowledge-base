# (16th place) Improve your imitation agent with rules

Competition: lux-ai-2021
Rank: #16
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/293835

[Github](https://github.com/tonghuikang/lux-ai-2021), [Notebook](https://www.kaggle.com/huikang/lux-ai-working-title-bot-private-version/notebook?scriptVersionId=81692756), [Download](https://www.kaggle.com/huikang/lux-ai-working-title-bot-private-version/data?scriptVersionId=81692756&select=submission.tar.gz), Submissions [105](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-24143846) [126](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-24165584) [129](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-24173911) [129](https://www.kaggle.com/c/lux-ai-2021/submissions?dialog=episodes-submission-24173878)

#### On sazuma's imitation agent

I would like to highlight some fine details of [sazuma's very influential imitation agent](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning).

- The center action is not being trained. If you see the sazuma's original bot, all units are "hyperactive" - they will make an action whenever they can.
- There is a "curfew". If the unit is in a city, and it is at night, the unit does not move. Because the units are hyperactive, they are likely to die at night.
- Limited unit vision. I made a [comment](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning/comments#1555995) that the action predicted does not take inputs from 15 tiles away. The unit is blind towards faraway objects, and may train based on correlation with other signals (the presence of edge or an opponent moving towards the location). nosound's UNet is one way to approach this problem. 
- The citytiles always produce units whenever possible. The citytiles are iterated in arbitrary order. Units are also ordered in arbitrary order.

The only change we made to the model is to add a center action and remove the curfew. syxming is responsible for the training. I may invite him to comment on how he approached the training.

These are the rules that I have applied on the imitation agent.


#### Building

- Sort the cities to encourage unit production at certain places. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/make_actions.py#L102-L111). However, the better approach is to learn the building the cities like what is described by nosound. Some rules can be applied here as well, such as giving building priority to the cities with the largest logits. Some rules could be used to consolidate logits to address the assumption of independence in inference - for example, if you are less likely to build a worker if you are already building one in the adjacent block.
- Abstain from building a unit if wood is to be conserved. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/lux/game.py#L434-L457). Again, this should have been learnt.


#### Movement

- Sort the units to encourage units resources with resources to act first. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/lux/game.py#L474-L481). The intention is that units without resources should give way to units with resources. I think there are other ways to sort
- If it is impossible for the city to survive in the next night turn, evacuate the units to jump out of the city. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/lux/game.py#L388-L414).
- Consider a unit in a city tile as occupied to prevent clustering. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/imitation_agent.py#L226-L227). Without this feature, units in a city can coalesce and stack up and waste unit count capacity.


#### Transfers

- If the unit is more making any action, and is not going to be full at the end of the turn, consider a transfer. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/imitation_agent.py#L204-L217). [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/imitation_agent.py#L204-L217). [code](find_nearest_city_requiring_fuel). I have explained the potential of transfers in [another post](https://www.kaggle.com/c/lux-ai-2021/discussion/278236).
- Add a bonus to the center logit to encourage transfers. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/imitation_agent.py#L204-L217). In the training, transfer actions are discarded (probably should not have). More bonuses are added for some tiles (uranium for example).


#### Rule overrides

- If the unit is far away from a resource and an enemy asset, let the rule base take over and return the unit to a resource. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/lux/game.py#L361-L381). For my final submissions, you do not see my units loitering at the corner and wasting unit count capacity.
- Send units with uranium to refuel cities. [code](https://github.com/tonghuikang/lux-ai-2021/blob/33f0709571d9ddc63f34f86c50640a2250c80953/make_actions.py#L608-L658). This discourages the concentrating uranium and wasting uranium to build a city. This is promoted by transferring uranium out of the resource tile, and a unit with substantial uranium but not collecting uranium will be considered for a "homing mission".
- Produce units at the end in case of tile in citytiles. [code](https://www.kaggle.com/c/lux-ai-2021/discussion/265767#1606884)

With these rules, the winrate against the center-trained curfew-removed vanilla imitation bot with the same model parameters is [around 90%](https://www.kaggle.com/huikang/lux-ai-working-title-bot-private-version?scriptVersionId=81692756#Evaluation), if there is no time limit.



#### What should have been done

Of course, redesigning the model to one that also learns where to produce a unit should give huge wins. These are the other aspects that could have been done better.


- My final bots exceed the time limit frequently, and I could not fix this in time. I hope the positive win rate can carry my final submissions to at least the high 1500s where I can see some of its performance against the 1650 bots. This is what I should have done
	- Limit the number of units further. I should have set up a limit the capacity to twice the number of opponent units.
	- Make only one inference of the model during testing time. sazuma's code reconstructs the matrix from the observation for each unit. I think it is possible to produce the predictions for all units with just one pass.


- Wasting effort. I spend quite considerable time writing and testing a rule-based only bot, only to find out that it has been made irrelevant. I did not foresee that reinforcement learning can achieve human performance, seeing how the top submission still make obvious mistakes until their submissions in the final weeks. This is what I should have done
	- Abandon the use of rule-base code completely after sazuma published the notebook, and probably retire from the competition. My initial attempts at training an imitation model were horrible and I gave up on not realising the potential of an imitation solution.
	- Apply more A/B testing and experimentation. Many hyperparameters could be tuned, but we did not have the time and compute to do.
	- Produce concrete results of the rule-based augmentation to make a stronger proposal for merger. A higher rating on the leaderboard would help. syxming and I tried to [invite](https://www.kaggle.com/c/lux-ai-2021/discussion/265767#1606884), but we did not receive any response.



#### Conclusion

I would like to thank the organisers for the well-designed game mechanics, sazuma for the groundbreaking imitation agent, my teammate syxming for improving the imitation agent, robga for the matches download scraper, nosound for some great ideas and Toad Brigade and RLIAYN for the data.

You can download my bot to play against yours with `lux-ai-2021` command line interface. There are some annotations visualise the rule based moves.

You can estimate the winrate of your bot against my bot with my [evaluation notebook](https://www.kaggle.com/huikang/lux-ai-agent-evaluation).
