# 9th place solution: RL with LookAhead, Floodfill, and Rules

Competition: hungry-geese
Rank: #9
Source: https://www.kaggle.com/c/hungry-geese/discussion/255931

I would like to represent my wonderful team of @robga @taahakhan @superant @returnofsputnik @khahuras to publish our solution, co-written by all members. Since it was a long 6 month challenge, we formed an early team so we could take breaks when life got in the way :-)

**Best Agent:** [github](https://github.com/digitalspecialists/hungrygeese)

Our solution was 7th with a score around 1230 on the day of submission closure and is 9th at the time of posting this solution description. We have 12 more days until finalisation. We hope to finish in the top 11 with Gold but it looks like we may end up anywhere from 7th-16th. 

Our approach is a synthesis of multiple approaches, with different agents having more or less weight from different aspects of:

- An RL value and policy network based on customised HandyRL
- A Look-Ahead capability to value potential board states several steps ahead
- Floodfill analysis to see where opportunity and traps lie
- A battery of hand crafted rules to manage danger and set aggression levels
- Imitation training to learn from the best of the leaderboard

What follows is an extensive explanation of our approach. We believe in the collaborative nature of Kaggle and encourage others to openly share their approaches and code, no matter their final position. We can all learn from one another.

## RL training (Rob @robga)
The core heart of our solution is based on the [HandyRL](https://github.com/DeNA/HandyRL) framework which provided a great foundation. We adapted the starter kit with changes to the hyper parameters, input shape, and network architecture. e.g. Additional layers included next step tails, food on two layers, time-to-disappearance gradients on snake bodies etc. Changes to architecture included a different pattern of residual skip connections. More drastic model changes didn’t show benefit.

Our RL training was on a single machine with TR3970x (32 core 64 thread) CPU, 128 GB RAM (64GB used), and 2080Ti GPU. We had periodic access to Google cloud with a V100 and 128 vCPU but this was surprisingly slower than the TR/2080. It culminated in 2 long experiments which took about a week each. Possibly more CPU/GPU/time would have led to a better result?

A significant action was to have thorough analysis of RL checkpoints which could vary wildly in their quality. Our evaluation competitor was the public HandyRL bot (‘HRL’) with 50 matches per checkpoint for a first pass, narrowing down the best to then have 1000, 10k, and ultimately 50k matches per checkpoint. Eventually, we could beat the Public HRL agent 72% of the time 1v3. If we add in knowledge of its deterministic moves, 100% of the time.

Here you can see the training curves of our two complete RL runs of 6 and 9 days. The black and blue lines show moving average performance, red and green show 2 std dev spread - there was high volatility epoch by epoch. The Y scale is our custom metric with 0 being the performance level of Public HRL. The eval agent is pure NN Policy with no additions. Our checkpoints could score 0.87 on our metric, which increased to 1.37 with Look Ahead, Floodfill analysis, and Rules.

[Training curve]

We were not successful with action masking or adding rules to the self play agents. This just seemed to prevent learning. In retrospect, we could have spent more time on introducing strong agents in the RL loop, not just self-play. We could have tried seeding RL with imitation checkpoints.

We had many features that work in rule based agents, like floodfill analysis and corridor detection. Frustratingly, we couldn’t get these to assist the RL trained NN when provided as concatenated input.

## Look-Ahead (Taaha @taahakhan)
Our lookahead (LA) setup is similar to the [Stockfish](https://www.chessprogramming.org/Stockfish) chess engine in nature. It is an iterative deepening recursive Alpha-Beta Minimax search with optimized speedup mechanisms. The network used in the heuristic was trained using the RL methods above with the HandyRL “value” output incorporated with flood-fill analysis. Loosely based on a previous [Battlesnake winner](https://github.com/m-schier/battlesnake-2019/blob/master/AI/AlphaBeta.cs).

**Lookahead Configuration:**
- [Alpha-Beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning) to cut branches that are guaranteed not to have better scores
- Custom move sorting based on collisions, food distance, opponent heads, etc.
- [Iterative deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) to search depth 1, depth 2, depth 3 etc. until timeout, `1.0 second per step + (remainingOverageTime / remainingSteps)`
- [Transposition Tables](https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning_and_transposition_tables) to save previous positions in iterative deepening for reuse. Dynamically sized to remove data from completed steps and quick lookup for future steps.
- [Heavy Caching](https://en.wikipedia.org/wiki/Cache_(computing)) to save previously computed data in helper functions for reuse
- Opponent Masking to remove opponents from the board if it doesn’t matter what move they make, this is slightly unstable, but drastically simplifies the search assuming the idea that if all body parts of an opponent are a certain threshold of distance away, there is no immediate effect to ourselves, so it doesn’t matter what move they make.
- Masking invalid & dumb moves to only the key ones. This cut the branching down to a more manageable amount.
- Rating collision moves slightly higher than a corridor death, so that its better to take down an opponent than to die alone
- Using Floodfill analysis to override the net score, mapping the flood fill space score from 0 to 6, to -3 to 3. This was added to the net score, giving values like 3.426 (at least 6 blocks of space and 42.6% chance of winning) or -2.002 (1 block of space with a 0.2% chance of winning). Floodfill is explained later in this post.

A problem arose that the LA would suggest moves that were generally not as good as the basic policy net, and the policy net would suggest moves in some cases that had more long-term problems than the LA moves. We needed to find the perfect balance of when to use LA and when to use the RL policy. I don’t think we quite got there, but we came to a close compromise. Only use the LA if it is strongly suggesting a move that the policy is misinterpreting, and use the policy if the LA is inconclusive between multiple moves, or if it has a confidence of >95%. This uses the policy more often, but falls back on the LA for key survival moves or clearly better moves in the future.

## Heuristics / Rules (Rob @robga)

Our hand-crafted rules are common sense and frankly not very novel, but they make a big difference.

**Danger spots and cautiousness**

If there is a spot we can move to that another goose can move to we call it a 'danger spot'. A potential head smash. A lot of game tuning was around this aspect. If you never move to a danger spot unless it is the last resort, you aren't using the board to its full potential.

Imagine a world where no goose uses danger spots. We can safely use them always. Now imagine a world where all geese use them. We'd better avoid them. Reality lies somewhere in between so it's about finding the optimum balance. Probably cautious and aggressive ended up in some sort of equilibrium. 

We decided to be highly aggressive most of the time and use those spots with abandon and only avoid them if

- it is the first 90 minutes of submission with kamikaze crazy bots around
- the opponent has no choice but to move there, once single-holes are excluded
- the spot has food
- the NN is very unsure to go there and we have another choice that has a very open space

**Traps:** A trap is when a move a goose takes turns another goose’s position into a corridor. Using floodfill and 1-step Look-Ahead we can spot traps. If we can be trapped on the next move, deprioritise the move. If we can trap another on the next move, prioritise the move.

**Corridors:** Using our FF algorithm, we strongly avoid corridors.

**Tail Head Food risk:** If a head is next to food, don't follow the tail.

**Dead on the next step:** If another goose is dead on the next stop, pretend they don't exist except for body collisions.

**PubHRL future:** Incorporate knowledge of the deterministic public HandyRl agent, which to this day is about 25% of all submissions.

**1v1:** Once it’s the final two geese, you should go cautious if shorter and very aggressive if the longest

Plus some ranking of what to do when presented with the above non ideal situations.


## FloodFill area scoring (Ant @superant)

While RL can learn to score a complete position, it can be useful to include human knowledge about the possible future of a position. The common [floodfill](https://en.wikipedia.org/wiki/Flood_fill) algorithm can be used to determine at which step a particular goose will reach a certain cell. This can be used to count cells which you reach before your opponent as your own influence area and can determine the desirability of a position.

We start with a position like 

Board:

[Board]

First we need to determine at which step (start counting from now=step 0), a tail segment will leave a cell that you can enter. We cannot know where a goose will go, and we do not know which goose will eat, but assuming no goose eats, we know when a tail segment will recede. Hence, we only simulate receding tails. 

Free on step:
 [free-on-step]

A value of 1 means, next step any goose can enter this field.
Assuming your goose is the red goose, we can find the step at which we will reach a particular cell if we go the most direct way and tails recede.

My floodfill:
 [my_floodfill]

We do assume that all heads stay where they are, which is not correct, but the best we can do. Later Look-ahead will take care of this in a better way. We can do the same flood-fill starting from all enemy heads. This will calculate at which steps _any_ enemy can reach a cell. Hence it is very pessimistic and assumes all enemies turn towards us. It would be possible to do this for each enemy individually, too.

All enemy floodfill:
 [all_enemy_floodfill]

What is probably most interesting now, is to consider how many steps we are ahead before an enemy at each cell, i.e. `all_enemy_floodfill - my_floodfill`. If this value is greater than zero, we know that we are guaranteed to reach this cell, no matter what the enemies do.

Floodfill difference:
 [floodfill_diff]

There are 13 cells with `floodfill_diff >= 1` and hence we could estimate that our guaranteed area is worth 13. Of course, the enemy geese cannot completely deny all red cells at once, but in the worst case they can deny many of them.

There are multiple options to use this information to guide a goose:
- You can use the whole `floodfill_diff` value field as a new layer input into a neural network. This will enable the neural network to understand goose movement better
- You can use `sum(floodfill_diff >= 1)` as a terminal scoring in a look-ahead tree search
- You can simulate “half a step” by moving our goose only into all directions, and calculate 4 numbers `sum(floodfill_diff >= 1)` for each of the directions. This can be passed as a length 4 fixed position input into a neural network.

Caveats and details of floodfill:
- Eating food is not taken into account. Using the distances to closest food and delaying tails according created more mess than good
- Our goose will happily follow other enemy’s tails and ignore the risk of them eating food while we have no way out. To prevent this, we could penalize enemy tails by increasing free_on_step barrier values every 3 enemy tail segments, i.e. assume the enemy will eat food somewhere.
- A goose can take a detour and hence reach a tail barrier later than if it went the most direct path. This would allow passing the tail, which floodfill with direct movement only would not have considered. Hence we implement a modified floodfill which crosses tails as soon as they become available, even though direct floodfill was at a location too early.
- It would have been nice to speed up floodfill scoring with Numba, however the dict structures were in the way. As the current floodfill is very fast anyway, we did not keep on trying Numba.
- An idea which we did not try was alternative scoring of the form `sum(f(my_floodfill, all_enemy_floodfill))`. One could try to model the area of influence by giving a small non-zero score for small negative values, i.e. leaking slightly into the enemy region. This may help, because the current `floodfill_diff` is very pessimistic. Additionally, one could consider individual enemies again and design a better score from that.

**Rule-bot attempt based on floodfill**

If we have a way to score positions, we can implement classical look-ahead tree search to simulate possible steps of all geese up to a certain future horizon. A full unoptimized depth-2 lookahead was easily feasible with floodfill scoring within the given time constraints.
To score positions we attempted the following compound score which is a 4 element Python tuple and allows for comparison. Most values were clipped to some maximum value to allow this part of the score to saturate, and activate the following tuple parts.
1. Number of reachable cells if enemy heads do not move = `clip(sum(my_floodfill != np.Inf), max=6)`
2. Number of cells we are guaranteed to reach before any enemy = `clip(sum(floodfill_fill >= 1), max=6)`
3. Length of my goose. This was needed to encourage eating
4. Minus distance to closest food (small is better)

Therefore, a goose will only go where it is not a dead end. Next, it will consider where it has enough undisturbed space. If all of this is fine, it will focus on eating.

This scoring rule with 2-step look-ahead, results in quite a reasonable bot. However, it becomes apparent that the board quickly gets very crowded and any direction the goose can go is possible death if the enemies would want to team up. 

Unfortunately, this made Hungry Geese more of a statistical survival game where made-crafted rules cannot be easily designed. Therefore, using the `floodfill_diff` layer as an input to a neural network and letting it figure out the proper weighting automatically by training is probably a more recommended approach.

Details of look-ahead:

Once you have an arbitrary scoring function and an implementation of common look-ahead tree search, there are not many free parameters anymore. One key part is how to _score deaths_, for which you will need to assign a well chosen score value.

Another key decision is how to _aggregate over the possible enemy moves_. As enemy moves happen simultaneously and enemies are not necessarily all adversarial, it is not obvious which operation to use. We tried `min()` (pessimistic), but `mean()` (enemies not adversarial) seemed slightly more successful.

## Imitation (Kha @khahuras, Corey @returnofsputnik)

Replays were scraped using [Simulations Episode Scraper Match Downloader](https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader). By the end of the competition, training was based on matches from submissions with scores of 1260+. We used a richer input and model architecture than we used in RL. A pure imitation based agent can score in the mid to high 1100’s. RL input/model changes were also trialled with Imitation, since the training is about 100 times faster.

Although an ensemble of imitation / RL was not a top tier local performer, we included it in our submitted flock for diversity. Our few imitation submissions without RL are now in the mid 1100s. This is an important strategy in kaggle simulations. Instead of only targeting a high final top submission, you can create a diverse army that competitors are going to encounter and perhaps be kept at bay with.

## Agents (Rob @robga, Taaha @taahakhan)
We put together the agents from a shared github repo, and slapped together all files of code into one file. This ended up to be about 1700 lines long of LA and heuristic code. The floodfill library was [sticky-taped](https://github.com/mwilliamson/stickytape) in for general usage. Models were stored separately in the tar.gz archive. The final LA ended on Version 017c, and the final RL network ended on Version 017k with 26 RL experiments and 657 submissions over the course of 6 months. We tried to maximize our submissions in the later months due to our experience in the [RPS Competition](https://www.kaggle.com/c/rock-paper-scissors/leaderboard) which was highly volatile in leaderboard score for the same agent, so more submissions of the same agents meant a larger chance of scoring high. There is a risk you get imitated, but this is smaller than the advantage of a large flock.

**OnnxRunTime**

We also made heavy use of [onnxruntime (ORT)](https://github.com/microsoft/onnxruntime) which reduced inference time to a third and opened up time for Look-Ahead. Not all our submissions relied on this saving, and some of our top submissions would have completed in time without it. But it made for much faster test and validation cycles. Using onnx means no torch import or operations were required in the agent.

As far as we can tell, this use of ORT to save kernel runtime has not been discussed on kaggle previously. We’d expect it to become widespread. We have added a [notebook](https://www.kaggle.com/robga/using-onnx-onnxruntime-in-submissions-for-4x-speed) to show the usage of ORT.

## Local Evaluation (Taaha @taahakhan, Rob @robga)

We had a robust setup to locally evaluate submissions to find a better and more consistent score for comparing agents. The game setup was 1 of our agents vs 3 copies of the public [HandyRL agent](https://www.kaggle.com/yuricat/smart-geese-trained-by-reinforcement-learning). We used a custom metric with scores `[2.0, 0.75, -0.75, -2.0]` for first through last place, summed up scores for N games, and divided the sum by N. We usually ran the evaluation for 1k to 10k+ rounds with multiprocessing to find the most accurate score. Our final agent scored about 1.379 on this metric, with win probabilities `[0.722, 0.139, 0.086, 0.052]`. This metric is with PubHRL detection Off.

## Code

- **Best Agent:** [github](https://github.com/digitalspecialists/hungrygeese)


## Final comments

I'm looking forward to champion level RL weights being posted to see how well our additions work on top of those.

Although we hope for the success of our geese and position in the final days, we wish the best to our competitors. It's been a fun challenge.

See you soon in Lux ;-)
