# 6th place solution summary

Competition: lux-ai-2021
Rank: #6
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/293776

It was a great competition, special thanks to Stone Tao @stonet2000 for making it happen. I am looking forward for any off-season activities and of course season 2. My solution is entirely imitation learning (IL), and multiple attempts at reinforcement learning (RL) did not succeed. In this summary I will first describe my IL implementation, and then share conclusions from my experience with RL.

The leaderboard will finalise in only 2 weeks (the lower limit), so I wish everyone who cares about their placement good luck with their final battles!


##The model

The architecture that I used I described in [this post](https://www.kaggle.com/c/lux-ai-2021/discussion/289540), below I attach an updated picture. There was a lot of discussion in the comments to that post, if one is interested in technical details.



One thing that I found is that it is very beneficial to fine-tune the last layers with replays from one agent. I usually started training with 7-20 top submissions, from different agents, then freeze most of the network and fine-tune the last 3 layers with data from "Toad Brigade" (TD) only. 

Two other components which I found to be very useful are learning transfer action and a city tile model. Regarding the transfer action, I am not sure if it is indeed very useful, or it just helps a lot because this is what TD uses, and I learn from TD. I tested different tweaks around transfer action, and the most aggressive use of it worked the best for me. City tile model was clearly beneficial for me too. It replaced a naive logic of building a unit if possible and research otherwise, but it turns out in many cases it is better to research instead of building a unit.

In the sketch above you can see that the model output is 7. The 7 actions are the following:
1. Unit: CENTER move action
2. Unit: NORTH move action
3. Unit: Build a city
4. Unit: NORTH transfer action (the first resource that found in cargo, all of it)
5. City tile: Build a unit
6. City tile: Research
7. City tile: Do nothing

Why only north for unit move and transfer? I used the trick that I described [here](https://www.kaggle.com/c/lux-ai-2021/discussion/290284). 

You can see I used the same model both for units and city tiles. And the general approach is the same for them, both for training and for inference I pick the 7 logits from the corresponding position of the unit / city tile that I am currently looking at. I really like that I need only one forward pass per turn. Even with just one forward pass the running time was getting close to the limits sometimes, so I guess per-unit approaches should really have struggled with the time limit.

Another important component is masking invalid actions. First, it is important to prohibit picking of clearly invalid actions, for example when the model proposes to build a city with only 98 cargo at hand. Second, a solid units collision avoidance logic is required. But then one can think of lots of different ways to improve here. For example the extreme here is blending imitation learning model with a rule based approach. The rule based proposes different missions, and the model picks the action with highest probability from the subset defined by missions. So basically masking all other actions, which appear as the first move in none of the missions. I tried some things but nothing interesting to mention.

During training I used one augmentation, - horizontal flipping of the map. When I need to decide to move north or to build a city, flipping the map east-west is a valid augmentation. I also used this flip as a test time augmentation (TTA), but it was giving very little improvement and sometimes (rarely) submissions with TTA were hitting time limits (therefore losing), so ended up using it in only half of the submissions.

##Reinforcement Learning

In this competition I worked more on RL approaches than the IL that I described above. I used @glmcdona implementation of the game engine and integration with stable-baselines3 [here](https://github.com/glmcdona/LuxPythonEnvGym). I mostly experimented with PPO method. I will not go into details, because I couldn't improve my IL with it, but this is what I learnt in general.

1. **Having an ability to imitation learn**, like in this competition, is important for RL success (didn't help me enough). First, RL is slower to train, so having an already pre-trained IL model saves lots of time. Second, IL allows perfecting the model architecture before RL runs. What if the model does not even have enough capacity to learn the concepts? One can test all of this with IL, become confident that the architecture is solid, and only then move to RL. Testing such things with RL itself is unrealistically expensive. 
2. **Simulate fast**. Having a fast simulation engine is important, because the RL training takes much more time than supervised learning. Implementing the engine in C should be considered. But at least take time at perfecting the infrastructure.
3. **Value function** should be as accurate as possible. It is even appropriate not to train the policy and the value networks at the same time. First train the value function with supervised learning on your agent replays, and then do PPO policy iteration with frozen value function. Repeat.
4. **Multi-agent environment**. This is the main reason why my RL failed, I think. My PPO was actually able to learn some correct moves with just a few units from scratch, but at some level of complexity it stopped improving. The problem is that the value function that I used is global, and therefore advantage calculation for each individual unit was extremely noisy, in presence of dozens or hundreds of other units. 
5. **Simplify**. Deep RL is hard. Start small, in this case, start from maps `12x12`. Maybe do PvE. Maybe even freeze the seed. But even *just* to learn to maximise final number of city tiles on a specific `12x12` map without opponent is hard, so be humble. 
6. **Tree search** approaches seem to be hard to implement here, because it is massively multi-agent. Interesting if someone succeeded.

##Feedback

I have participated in quite a few Kaggle competitions by now, and I can say I am still to meet a competition which is organised without a glitch. So here we go:
1. The LB match making, scoring and final submissions policy are BAD. It was discussed in many threads, here and on discord, lots of room for improvement.
2. There were some rules changes in the beginning, the very important python engine took time to take off (and still with small bugs actually), and other problems with utilities here and there. This can be of course explained by the fact that this is the first season, and the organisers deserve the highest mark for handling all issues fast and professionally.  
3. Working with `meta-kaggle` dataset is not a good experience. It is huge, and for example right now has not been updated for 5 days already, despite promised daily updates. 

##Finally

This competition is awesome, unique and addictive. The replays are great, not sure how many hundreds I have watched. In memory of all the units that I sent to work at night:

