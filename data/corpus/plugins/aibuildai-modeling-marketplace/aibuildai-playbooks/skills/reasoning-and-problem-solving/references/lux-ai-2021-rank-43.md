# 43rd place and my best local tested approach

Competition: lux-ai-2021
Rank: #43
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/294070

Update after finalized leaderboard:

The 43rd place model was trained with Team Durrett replays and with added center action from their games in contrary from the others, but all other techniques described from the approach here was included. 
All in all, for the competition, it ended up with 54 silver agents trained and tuned with a combination of described approach below.

--------------------

Looking at the past competitions’ winner’s reinforcement learning (RL), rule-based and imitation learning (IL) are the way to go. In previous competition I put my effort in the RL cup, willing to learn more in that area but in this competition, I took a 50/50 approach from start, with a goal to use the IL as one of the agents in the RL training phase. In short this didn’t go better than the IL itself, so left the track halfway, this time.
For the IL I used @shoheiazuma code as a base, looking in the past this will give a good starting block. 

**Data**

Having a good replay version dataset control was one key, to easy sort the latest replays and upcoming agents without had to download massive of data every time.
I early started to test different filters with the data, different thresholds for the final total citytiles of the replay, map sizes etc. but leaved it as-is with only a different filter to use in different trainings, only-win and both win-lose filter, also saw that some doesn’t show center moves at all in replays while some does, valuable when training different models.

**Benchmark**

During the race I did local benchmarks before releasing the agents, this after noticed that an agents first indicative score is at the best after 2 weeks in the public leaderboard. 
I used a local leaderboard to benchmark different solutions and agents with different map sizes, and the combine the best agent for a specific dimension to one single agent. While doing this I also trained agents for only one map size with filtered data with only that size, the result was better than the original but not always, depending of course on the amount of data. I left this approach early, minimize the testing horizon, maybe I should with more time tested it again at the end now with more data. But I continued the benchmark choosing the best model vs map size but trained with the complete data.

**Best local approach**

**Common for all models in the agent:**

- Trained 20 epochs with TTA (vflip,hflip,hvflip)
- Adam
- CosineAnnealingWarmRestarts.
- Data from 10 okt-01 dec replays over 1750 score
- Mixed Precision/AutoCast
- Clips gradient norm. 
- As the data was too large, I used many resumed trainings and only trained with 10k steps per epoch with random batch. 
- All model converted from Pytorch to ONNX runtime for faster inference, a technique learned from past AI game challenge and studying @robga top solution, way better than quant or float16 mode and no need to load the torch framework, saving time in this 3 second 1 CPU environment/frame.
- No general rules besides unit.can_act() and added a rule that keep track of the timelimit. With more than 6 seconds left of the 60 second limit I use the full solution and under 6 second I used only the smallest of the models, this way I never got a “submission error”.

**For the 32x32 dimension and map size:**

A smaller and faster architecture trained with both the top agent (TB) win games + lose games but use the win teams actions, this with TTA inference+mean values (also tried max value) it gave a boost to the dimension. I also used threshold for taking actions otherwise keep it centered. In the training I also use “Robust Bi-Tempered Logistic Loss Based on Bregman Divergences” https://arxiv.org/abs/1906.03361 to handle noise in data and smoothing across the labels.

**For the 24x24 dimension and map size:**

Same as for 32x32 but a larger model without TTA inference.

**For the 12x12 and 16x16 dimension and map size:**

As said earlier I noticed that not all submissions gave center actions in replays, but the third-place agent did " Team Durrett”. Here extended the labels from 5 to 6 and added the center action. It was the same architecture as the larger 24x24 model but as the center actions gave an extreme imbalanced label situation the label smoothing was not enough, so I instead used CrossEntropyLoss with Weights to the labels, balancing the training.

I have also submitted an agent with TTA for all above models over 6s limit else the smaller one, will see how that goes in the end.

**That’s it!**

-----------------------

I hade some more approaches but this solution was the best local, but maybe the final will be different, so maybe more to come.

Fun extra stuff:

I managed to use a custom efficientnet_lite0(change of pooling, bottleneck, activation etc.) with a forward pass resizing bilinear of maps/dims to 128x128. This gave a model in the medal zone with just 6 epochs, and I saw the higher the dims the better score but due to non-GPU kernel this was not manageable here.
