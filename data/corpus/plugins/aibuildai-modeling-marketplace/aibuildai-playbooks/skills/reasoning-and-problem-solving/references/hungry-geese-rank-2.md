# Goosebumps' solution - 2nd place

Competition: hungry-geese
Rank: #2
Source: https://www.kaggle.com/c/hungry-geese/discussion/263686

# Introduction
First of all, we would like to thank the team at Kaggle for hosting such a fun competition, and the many competitors who shared their ideas through tutorial notebooks, starter agents, and strategy discussions. An additional thanks must be given to HandyRL ( @yuricat and @kyazuki ) for open-sourcing their reinforcement learning framework early on and helping to introduce so many to reinforcement learning. Kaggle’s excellence manifests in its wonderful welcoming community, and this competition was a prime example of that.

Our team took a two-pronged approach to agent design, where @pressman1 worked on applying varied deep learning techniques to the problem, and @lpkirwin worked on various hand-crafted agent designs accelerated with Numba and Rust. @vishyvishal ended up being busier than anticipated, but was nonetheless a fantastic help in the brainstorming and hyperparameter optimization phases. While we were unsuccessful in explicitly integrating the different strategies into a single submission, our approach allowed us to each work on what interested us most, share ideas and suggestions, and concentrate our efforts on the most successful agents as the end of the competition neared. Our best agent ended up using deep learning for behavioral cloning in conjunction with Monte Carlo Tree Search, plus a few custom modifications to account for various complicating factors specific to this competition. For this write-up, we’ll not only go over what worked for our best agent, but also cover some of the other techniques that we tried, as many of them were quite competitive with our best agent. Suggestions are welcome - we are all here to learn!

# Deep Learning approaches (Isaiah @pressman1)
All code for the deep learning approaches can be found on [GitHub](https://github.com/IsaiahPressman/Kaggle_Hungry_Geese). For deep learning, we were working with an 8C/16T dual-GPU machine with 32GB RAM. This was plenty of horsepower for supervised learning and supervised learning adjacent applications, but limited our capacity to run traditional CPU-intensive reinforcement learning at scale. This influenced many of our decisions about how to design and train agents, and which research directions to pursue.

### Our best agent
The best agent ended up using deep learning for behavioral cloning in conjunction with Monte Carlo Tree Search. Additionally, we made a number of modifications to account for test-time hardware limitations, decision time limits and overage time rules, and opponent unpredictability. 

### Behavioral cloning
In behavioral cloning, a deep neural network was trained using replay data to, at each step, predict the moves of each player and the final outcome. Cross entropy loss was used for the policy prediction and mean squared error loss for the value prediction. A large weighted entropy penalty of 0.1 was added to the policy loss to incentivize greater uncertainty in the predictions. Additionally, for data augmentation during training, we performed random horizontal and vertical reflections and randomly permuted the order of the agents. Both of our model architectures were translation equivariant on a torus, so no translation data augmentation was necessary.

### Data selection
Each day, we downloaded replays from the top agents. (Thanks @robga for the tutorial notebook) We only downloaded and trained on episodes wherein the worst agent’s latest leaderboard score was greater than some threshold t. We increased t over the course of the competition as more stronger agents were submitted, and an increase in t almost always lead to a notable increase in the trained network’s strength, so long as there was still enough training data. Our motivation for this data selection structure, as opposed to one where all replays from the top N agents are used, was to minimize the amount of bad data fed to the network. Lower rated agents will often make poor decisions and have unpredictable game outcomes. Our training algorithm did not try to distinguish between the good and bad moves or better or worse agents within a game, so it was important to minimize the number of poor decisions and unpredictable games in our training dataset.

### Input encoding
For the model input, we encoded the board as a 19x7x11 tensor. The first 16 channels consisted of 4 channels for each agent:
* Contains head - a matrix of all 0s with a 1 at the agent’s head.
* Contains tail – a matrix of all 0s with a 1 at the agent’s tail.
* Contained head last turn – a matrix of all 0s with a 1 where the agent’s head was last turn.
* Contains body – a matrix of mostly 0s, except where the agent’s body is. Values of 1 / 100 represent the tail and n / 100 represent the nth body segment, counting from the tail and including the head and tail.

The 3 additional channels encoded the location of the food, how many steps it had been since the last starvation, and the current step.

### Model architecture
We used two different base model architectures, and experimented with a few different `n_block` - 4, 6, and 8 - and `n_channel` - 64, 92, and 128 - configurations for each. For the convolutional architecture, we stacked `n_block` squeeze-excitation residual blocks, each consisting of 2 3x3 convolutional layers with `n_channel` channels, circular padding, ReLU activations, and no normalization. For the attention architecture, we stacked `n_block` residual blocks, each consisting of a 4-headed self-attention layer with torus relative positional embeddings followed by a 2-layer perceptron, all with `n_channel` channels, GELU activations and layer normalization. Ultimately, our strongest model was convolutional with 6 blocks and 64 channels. However, 4 block/64 channel convolutional models and 8 block/128 channel attention models were also very successful.

Both base model architectures preserved the 7x11 input shape, so for the output layer, we indexed the hidden representations of each living goose head, producing 4 vectors with `n_channel` values. We then applied 4 separate multi-layer perceptrons, 1 for each goose, to reduce the size of each agent’s policy output to 4, followed by a softmax layer, producing a 4x4 output policy matrix representing the move probabilities for each goose. For the value output, we also applied 4 separate multi-layer perceptrons, this time each outputting a single value. We then applied a softmax operation to the 4 output values to normalize them into the range ([0, 1]), and then projected those values to the range ([-1, 1]). Notably, the value outputs are therefore interdependent such that they will always sum to 0, which would not necessarily be the case if we simply applied a tanh activation to each value output.

[Model architecture]

### Monte Carlo Tree Search
The base model’s predictions are not very strong on their own, because the model is not trying to predict what to do to win, but rather trying to predict what the average agent from the training set would do in a situation. If the average training agent would make a mistake in a tricky situation, then the model will be trained to also make that mistake. To improve upon this, we used the PUCT MCTS algorithm - Monte Carlo Tree Search with Polynomial Upper Confidence Trees - as was done in AlphaGo. For each iteration of search we selected an action for each agent based on the PUCT formula until we reached an unvisited or terminal state, performed inference on that state to generate policy priors and a value, and used the value to update the Q-values of the actions that had been taken. For the search policy, the probability of action (m) being best was computed as:

$$
p_m = \dfrac{n_m}{\sum_{i=1}^4n_i}
$$

where (n_m) is the number of times that action (m) has been taken during search.

We did not use any compiled libraries or ONXX to speed up the tree search or neural network inference, so we performed around 20-50 iterations of search per second on the Kaggle servers, depending on the model size. Additionally, some adjustments we made to the search included:

* We used action masking to ignore actions that would result in guaranteed immediate death.
* We did not include the location of food when considering whether a state had been visited by the MCTS algorithm. This way, we increased the depth of the search at states where a goose was about to eat, but introduced bias into the returned policy and value priors, as only one possible food spawn location was considered for inference each step.
* We considered the direction the search seemed to be heading when selecting the final action to return. For example, if an agent’s prior policy was ([0.25, 0.75, 0, 0]) and it’s search policy was ([0.4, 0.6, 0, 0]), we selected the less likely action A instead of the more likely action B, reasoning that such a drastic policy shift in few iterations would likely have converged to action A if given enough time. We defined a “drastic policy shift” with a hyperparameter delta, which we set to 0.12.
* We dynamically used overage time when the search result was uncertain. If after performing 0.93 seconds of search, the prior policy had shifted some amount towards a less likely action, but not by at least 0.12, for example from ([0.25, 0.75, 0, 0]) to ([0.3, 0.7, 0, 0]), then we continued searching for another 0.5 seconds. We repeated this process until one of four possible outcomes occurred:
   * The policy shifted by more than 0.12 in favor of the less likely action, in which case we selected that action.
   * The search policy shifted back such that the best action probability proposed by the prior policy had increased, for example, from ([0.25, 0.75, 0, 0]) to ([0.3, 0.7, 0, 0]) and then to ([0.2, 0.8, 0, 0]), in which case we selected the more likely action.
   * The search had used up 3 seconds of overage time and was still uncertain, in which case we selected the more likely action.
   * The agent dipped below 2 seconds of remaining overage time for the game, in which case we stopped using overage time and only made decisions using the results of the initial search.

### Learning opponent unpredictability
Our algorithm suffered when the opponents acted unpredictably, and even our best agents would too often place 3rd due to head-on collisions with a shorter goose that the neural network predicted would never move towards us. To counteract this effect, we computed the policy predictions for each opponent as:
$$
\pi_i = (1-w_i) \cdot \hat\pi_i + w_i \cdot \begin{bmatrix} 0.25 & 0.25 & 0.25 & 0.25 \end{bmatrix}
$$
where (\hat\pi_i) represents the initial neural network predictions for opponent (i), and (w_i) represents a learnable weight parameter for that opponent. At each step we tracked the initial policy predictions and the move the opponent selected, and after each step we used all previous steps as a dataset with which we performed a few iterations of gradient descent to update (w_i) to minimize the KL divergence of (\pi_i) with respect to the selected actions. This way, we added noise to the policy of unpredictable opponents, allowing us to better avoid otherwise unexpected head-on collisions, but still treated predictable opponents normally. 

### Other deep learning approaches
* We tried self-play reinforcement learning with A2C and Impala by running the environments on the GPU. This was incredibly efficient, and allowed us to run hundreds, and sometimes thousands, of environments in parallel on a single 8GB GPU, resulting in 5-6 games played per second. It was satisfying to train a competent agent within an hour, but ultimately the policies yielded by this approach were too close to deterministic to be combined usefully with our MCTS algorithm, and so the resulting agents were weaker. Given more time, I would have liked to pursue this direction further using larger neural networks.

* We tried a few ways of combining the policy of the neural network with a much faster compiled search function, (see @lpkirwin’s work below) but found the results to be weaker than either method on its own. Given that this is what HandyRL did in the end with excellent results, we hypothesize that this may be because their self-play network made better initial predictions than the behavioral cloning net, which rarely made it to step 200 when playing against itself without search.

* We tried an AlphaZero style approach of including MCTS in the self-play training loop. This kind of worked, but was too slow to train.

# Hand-crafted approaches (Liam @lpkirwin)
Unlike Isaiah, I did not have any beefy machine kicking around to train RL agents, so I focused on pure handcrafted MCTS agents. These agents topped the leaderboard in the early competition, but steadily fell away as RL competitors improved.

When I checked earlier today, my top rules-based agent had a score of 1189, which would put it a little outside of the top 10. I made very few changes to the agents in the last three months of the competition, so I think it’s likely that with more work, a pure no-ML agent could have stayed in gold medal territory.

The main strength of these agents was in doing a very deep search, which allowed the agent to avoid many traps that were 10-20 steps ahead. This was particularly true in the endgame, when few snakes and lots of forced moves dramatically reduced the search space.

The main weakness was that the agents’ algorithm would often badly mispredict opponent moves. The MCTS would be very confident that an opponent would move in a certain direction, and pick the optimal conditional move – and then the opponent would do something completely different. Unlike ML approaches that could systematically learn from actual opponent behaviour, I was only able to hand-tune parameters to try to mimic what I saw in online replays.

### MCTS structure 
I started out by trying to follow the AlphaZero MCTS structure as closely as possible - see p.15 [here](https://science.sciencemag.org/content/sci/suppl/2018/12/05/362.6419.1140.DC1/aar6404-Silver-SM.pdf) - but relying on a simple heuristic value function instead of a neural net. I went on to make several modifications to the algorithm to better suit the game:
* Hungry Geese is stochastic in terms of food replacement while many games usually played by MCTS agents are deterministic. Like other competitors, I choose to represent tree nodes only in terms of actions. I brought randomness in by always simulating states from the root node and randomly placing food as I moved down the tree. During the search I might pass through the same future state hundreds of times with various food placements. The idea was that the value scores from different food placements would average out into a roughly correct score for each branch.
* I added risk aversion: instead of searching based on mean scores for each branch, I searched using a weighted average of the mean score given the expected distribution of opponent moves and the lowest possible score that could result from going down that branch if opponents made moves that were as bad for us as possible.
* When doing the tree search, I had two separate exploration terms: one made sure that each agent was adequately exploring its individual move choices, the second made sure that different combinations of agents' moves were being equally explored. I got a fairly big boost in performance when I put this in.
* Instead of priors from a NN, I had very simple priors that made instant-death moves very unlikely, and moving to food slightly more likely.
* My heuristic value function was essentially:
   * Final scores are ([-1.0, -0.33, 0.33, 1.0]) for last to first, respectively
   * If a snake has died, it gets its ‘final’ score
   * If a snake is alive, it splits the remaining available points proportionally by length
* I experimented with ‘flood fill’ value functions that were similar to those described in other writeups, but these were much slower to evaluate - so fewer future states explored - and actually made the agents a bit worse
* I experimented with several different ways of making the search more efficient, for example by sometimes recursively exploring a deep node rather than always resetting at the root. Sometimes this helped, but it wasn’t a killer feature.

### Numba, then eventually rust
The biggest part of my time in this competition was spent trying to make things go fast. My first implementation of the agent was in numba. I really do like numba – it’s quick and easy to plug into surrounding python code. Putting the game logic and tree search in numba allowed me to evaluate about 30-40k future states each second.

One problem with numba was the JIT compile time. While my agent would compile in about 10 seconds locally, it would often take 45-70 seconds to compile on Kaggle’s servers. Of course, if it went over 60 seconds, I was out of overage time and would error out. I found that, in particular, the ‘test episode’ servers were reliably slower than the other evaluation servers, so I built in some behaviour to disable JIT compilation for the first 15 minutes or so after submitting a new agent.

Later on, since I had been meaning to learn rust for a while, I took the opportunity to do that by porting the existing numba agent to rust, with very few other changes. This resulted in approximately a 10x speedup over numba, allowing me to search about 300-400k future positions per second. And since rust was pre-compiled, there was no more JIT overhead! 🎉

Unfortunately, there were only modest gains to agent performance. The strength of the deeper search was in finding forced wins and dead ends that were much further in the future – but this was already what the agent was good at! More speed didn’t do anything to solve the fundamental problem of mispredicting opponent moves as described above.

### Testing hand-crafted agents
Evaluating hand-crafted agents was a huge pain. In theory you can run a local tournament with all your different agent versions - but since MCTS agents use every millisecond of search time they can get, evaluation is super slow.

My strategy to speed this up was to watch online replays and manually collect scenarios where my existing agents made clear mistakes like walking into a dead end. I would then code these up as test cases that my new agent was expected to pass. For example, a test case would say: “download episode 123456789, go to step 154, evaluate the next move of the red snake, and fail the test if the next move isn’t south.”

This approach wasn’t very clean or automated, but helped me make sure that I wasn’t actively making the agent worse as I added new functionality.
