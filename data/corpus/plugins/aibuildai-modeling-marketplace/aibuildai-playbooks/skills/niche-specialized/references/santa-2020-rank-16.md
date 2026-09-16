# [16th place] A pure self-play reinforcement learning approach to the Santa 2020 challenge

Competition: santa-2020
Rank: #16
Source: https://www.kaggle.com/c/santa-2020/discussion/217020

First off, I want to thank Kaggle and the Kaggle community for this challenge. I have had a great time over the past couple months thinking about and working on this game, and have derived great joy from its combination of simplicity with surprising depth. Initially, I intended to work on this problem the classical way by designing an agent by hand, and only tinker with RL once I had achieved a satisfactory performance. However, after spending some time working on an algorithm only to have it promptly crushed by a [public notebook](https://www.kaggle.com/a763337092/pull-vegas-slot-machines-add-weaken-rate-continue5), I changed plans and committed myself to getting as far as I could using a purely deep reinforcement learning approach, at which point my teammate joined in.

## Principal challenges for an RL approach

In applying RL to this problem, we encountered four main practical and theoretical obstacles:
- Limited hardware resources
- The best features to provide as the observation is not obvious
- The best way to shape the reward function is not clear
- The environment is not modeled well by a standard feedforward neural network, at least not in it's most basic form

### Working with limited hardware resources

We did not have access to a compute cluster, nor a large budget for cloud computing, so the first priority was increasing the efficiency of training on a single-GPU home desktop. Using the environment as it was originally implemented took ~3 seconds per episode per process, not including time spent performing inference or gradient descent. While this is an acceptable rate of play, there is significant room for improvement. The environment can be rewritten by representing the thresholds as a matrix of shape \\(nEnvs \times 100\\) and the observations for both players could be returned as a tensor of shape \\(nEnvs \times 2 \times 100 \times nFeatures\\). This allowed us not only to run many environments in parallel, but also to run the parallelized environment itself directly on the GPU, eliminating CPU-GPU transfer time and efficiently perform inference and gradient descent on the GPU as well. The importance of this speedup cannot be overstated, and I believe was one of, if not the most, crucial element for us applying RL successfully. To get a sense of the scale, consider that running a 10,000-game match between two naive Thompson Sampling algorithms took over 30 minutes using the multiprocessing approach, compared to 12 seconds for the GPU-parallel approach - a more than 150x speedup, which was only compounded further by the GPU acceleration of neural network inference and backpropagation.

### Defining the observation space
- The first and most basic observation type that we tried consisted of a vector of 3 features per bandit: the number of times the agent had selected that bandit, the number of times the opponent had selected that bandit, and the sum of reward received. We also found empirically that including a fourth feature, the current timestep, helped to accelerate and stabilize training, despite being theoretically unnecessary and already included in the first three features.
To our surprise, despite clearly lacking important temporal information, this observation alone was descriptive enough to result in quite high performance. As of writing, 7 out of our 10 best agents use this observation.
- Another observation type we tried was a \\(1999 \\times 3\\) matrix representing the full 1999 timesteps of my_pull, opp_pull, and reward information. We tried using an LSTM directly on the sequence, and and a feedforward network on the flattened observation, a vector of length \\(5997\\), but had success with neither. The LSTM was not stable during training, and the feedforward network was too slow to train. Given another month, this is the approach I would have liked to explore more, particularly using the LSTM, as it should theoretically be well-suited for the task.
- We tried including the summed observation alongside the information about the last 10 steps, with the thinking being that the summed observation mostly captures long-term information, and the last 10 steps should capture some immediately pertinent information well, such as when the opponent is repeatedly using the same bandit. However, we did not find that the performance using this observation type meaningfully exceeded that of **1.**
- Finally, we tried providing an observation of shape \\(60 \\times 4\\) for each bandit, where each row in the matrix represents an *event,* defined as any timestep where that bandit is pulled by either the agent or the opponent. Each event includes the standard my_pull, opp_pull, reward information, alongside the timestep that that event occurred at. 60 was chosen empirically as <1% of high-level games have more than 60 events for a given bandit. This observation was then flattened and given to a feedforward network, and was able to train significantly more efficiently than the full \\(1999 \\times 3\\) matrix, as there were fewer parameters to train and the inputs were less sparse. As of writing, this is the observation space used by our top agent.

### Shaping the reward function

In theory, the best reward function should consist of simply 1/0/-1 at the end of the game, given possible results of Win/Draw/Loss. However, in practice, this resulted in an overly sparse reward signal that slowed down training. At first, we had success using a reward function that gave the agent a reward at every timestep equal to the expected value of the bandit selected: $$EV_{pull}=\frac{\lceil selectedThreshold \rceil}{100}$$ However, this reward function optimizes for greedy behavior, and does not encourage or account for any sort of deception. Therefore, we eventually settled on the zero-sum version of the previous reward, where the agent receives at each timestep: $$EV_{agentPull} - EV_{opponentPull}$$

### The problem with the Perceptron

The easiest way to feed the predictions into a neural network would be to take the raw observation matrix of shape \\(100 \times nFeatures\\), flatten it into a vector of length \\(100 * nFeatures\\), and use a standard multi-layer Perceptron from there. However, this approach is fundamentally flawed, as each bandit's features are given their own weights. Therefore, if the agent happens to have have good luck with a given bandit during training and bad luck with another, at test time it will favor the "lucky" bandit even before the game has begun.

Another approach would be to apply the same network to each bandit individually, and then apply a softmax operation at the final layer to select a bandit. Unlike the first method, this approach will remain unbiased towards any particular bandit, but suffers from the problem that the network as a whole is at best simply estimating the EV of a given bandit and then taking a softmax over some rescaled version of that EV, thereby ignoring all contextual information.

What is needed is a network architecture that is both unbiased towards any particular bandit, but also able to consider the features of all bandits simultaneously when processing the observation. In other words, we need a neural network that operates on sets. We tried a number of different designs, and had the most success with a network where each layer has the following form, assuming a \\(100\times n\\) input matrix \\(X\\) and a \\(100 \times 2n\\) intermediate matrix \\(M\\):
$$For\ each\ bandit\ b,\ row\ m_b = \begin{bmatrix}x_{b,1} & x_{b,2} & ... & x_{b,n} & xOther_{b,1} & xOther_{b,2} & ... & xOther_{b,n} \end{bmatrix}$$

where \\(xOther\\) is the mean of that feature over all other bandits
 
$$xOther_{b,n} = \frac{1}{99}\sum\limits_{i=1\ i \ne b}^{100}x_{i,n}$$

Then, using a learnable weight matrix W of shape \\(2n_{in} \times n_{out}\\), we get the \\(100 \times n_{out}\\) output matrix Z

$$Z = M \cdot W$$

For each intermediate layer \\(l\\), we take \\(ReLU(Z^{l-1})\\) as the input. For the final layer, with \\(n_{out}=1\\), we can take \\(Z\\) as the Q-values for Q-learning methods, or \\(softmax(Z)\\) as the probability distribution over the actions for actor critic methods.

### Miscellaneous thoughts

- We tried a few different model architectures, and had the most success with 8-layer residual networks, with four hidden layers of size 64, four hidden layers of size 32, and then the output layer, but this architecture is almost certainly an area for improvement.
- The RL algorithms that we had the most success with were A2C and AWAC. Q-learning did not work well. The biggest disappointment was being unable to get NFSP to converge.
- A self-attention method should have been at least as good as the layer design that we ended up with, but we could not get it to work as well. This may just have been due to memory/computation constraints.
- It was surprisingly difficult to predict any given trained agent's success against the field. Frequently we found that, using some innovation or another, trained agents could defeat all previous best agents and public notebook agents with large margins of 55% or more, only to do worse against the field as a whole when submitted. This all despite never being explicitly trained against said previous or public agents. We suspect there is some sort of strategic rock-paper-scissors effect going on, and perhaps this could have been avoided given a more diverse set of handcrafted validation agents.
To counteract this issue, we ended up just submitting somewhat trained agents  - >50% winrates against the internal validation field - ad hoc until we hit upon one that seemed to be doing well, and then continuing training for those agents. However, there is almost certainly a better way to validate agents than this, so any insight into how others performed internal validation and/or prevented strategic overfitting would be greatly appreciated.

### Conclusion
Each successfully trained agent was usually able to convincingly defeat all previous ones, and maintain or improve upon the public leaderboard standings, so given enough time and compute, we suppose that this self-play method has a much higher ceiling than what we achieved. This speaks further to the surprising depth of the game, as we would not expect continual improvements to come so readily for a simpler game. Any thoughts or feedback on our methodology would be greatly appreciated, and we look forward to the next simulation challenge!
