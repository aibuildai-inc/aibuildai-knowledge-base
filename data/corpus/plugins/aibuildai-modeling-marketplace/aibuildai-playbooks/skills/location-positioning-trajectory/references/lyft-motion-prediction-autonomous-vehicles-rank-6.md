# 6th place:  Micro-inputs, Lots of Data + Distance Order to Ensemble!

Competition: lyft-motion-prediction-autonomous-vehicles
Rank: #6
Source: https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/199588

First up: great competition! The data was about as stable as I’ve ever encountered on Kaggle – kudos to the organisers. Congratulations to all of the winners - and to everyone who worked hard and completed the competition. 

It was a pleasure to work with @rytisva88 and @sheriytm on this – thank you both! 

I’m sure that my teammates will post separately, so I’ll stick with some observations from my own workflow here. Very interested to hear how everyone else approached the problem, as there was a lot of scope for different approaches! 

I ended up with two small input sizes in the interest of speed: 128 + 5 channels, 196 + 5 channels. 
- The 128+5 model got to 11.9 after 68.8M samples. 
- The 196+5 model reached 11.15 after 71.4M samples. 

Reading others’ comments and solutions it seems that some of the items that I thought were key to doing well were not, in fact! I’m sure cherry picking ideas that worked from different teams could lead to something very interesting.


**Data, data, data:**

There seemed to be one major key to success in this competition: how much data you could access, and how you sampled it. 

Early on it became clear that grouping scenes while training gave a significant uplift: when a single agent from each scene was selected before moving to the next iteration, scores improved.

Taking this idea and following it through to train_full.zarr gave the next breakthrough: training on a chopped version of train_full.zarr yielded further improvements. 

The next jump in performance came from training on multiple chops of train_full.zarr. (You can adapt the l5kit code to create lightweight chops comprising just a small number of historic frames. This allowed for big savings on RAM/disk space.)

We debated why there were such performance gains from training using chopped versions of train_full.zarr rather than randomly accessing indices. From early experiments it seemed like ensuring scene diversity was important: in the same way that we might enforce class balance while training, enforcing scene diversity (and perhaps more fundamentally, driver diversity?) seemed to matter. 


**Raster size v data coverage tradeoff:**

Covering as much data as possible mattered, and training for as long as possible mattered. A single chop of train_full.zarr (approx 825K samples) could be shown to the model 12 times before it stopped learning. Obviously if you could show the model different samples you would do a lot better! But this seemed to be the hard limit.

Keeping the model as small as possible meant it could iterate through these samples much more quickly. The inputs for the models that I ended up using were raster size 128 and 196, history_num_frames = 5, condensed into five input channels: 

sum(agent_history), agent_current, sum(ego_history), ego_current, sum(semantic_map)

I was originally using Resnet18, but then switched to Resnest50 on @rytisva88's recommendation and it gave an improvement of about -1 in nll. (Incidentally, @rytisva88 had the best performing single model in our group).

Summing the semantic map meant that red and green traffic signals were treated the same. This seemed to work fine, surprisingly. Perhaps because only yellow gave additional information not already contained in the traffic movement.


**Acceleration:**

Eyeballing the predicted trajectories, it became clear that the models were ultimately making a bet on acceleration: typically, modes 0, 1, 2 represented trajectories arising from different agent speeds. 

Once it became clear that this was key it was possible to look at sampling the data such that we balanced these cases. 

A scene containing lots of agents was indicative of traffic. Gridlocked traffic obviously does not move much and leads to a lot of duplication in inputs. The models implemented a sampling scheme whereby scenes with a large number of agents were undersampled. The sampling proportion was: min(1, 7/agent_count). Thus, scenes containing 14 agents had 50% of those agents selected for each training iteration, etc.


**Ensembling:**

This was initially a tricky one: averaging models based on confidence values didn’t work. However, when the importance of acceleration was taken into account, the route to ensembling made sense: order the model modes by distance covered, then average the results. When this was implemented all models could be ensembled very quickly, with positive results. We also looked at incorporating curvature here, but it didn’t make any difference: distance was the key.

The final ensemble optimized weights on the validation set. The weighting scheme incorporated distance and confidence values. 


**Code:**

Code for contribution to our team solution can be found [here](https://github.com/ciararogerson/Kaggle_Lyft)


**Ideas that didn’t work: many! Here are a few...**

_Traffic lights:_

I couldn’t get additional traffic light information to add anything: many different angles were tried! The most promising was probably traffic light persistence: we included an additional channel in the model where instead of traffic light lane lines, we drew lines containing the number of frames since the traffic light had turned to its current colour (up to a maximum of 100). Ultimately this didn’t add anything.

_Day/hour:_

Adding channels for day/hour values proved better than concatenating them directly before the dense layers of the model, but it still didn’t help much.

_Interpolation:_

Having the penultimate model layer output 25 sets of (x, y) points, followed by an interpolation between these points to make the final 50 point trajectory did not work.

_Weighted loss function:_

Given the importance of capturing acceleration, we tried a version of the loss function that weighted  the last 10 points of the trajectory equal to the first 40. This didn’t help.
