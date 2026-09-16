# 8th place solution: Deep Neural Networks & Tree Search

Competition: lux-ai-2021
Rank: #8
Source: https://www.kaggle.com/c/lux-ai-2021/discussion/294603

## Motivation

+ The purpose of my participation in this competition was to learn application development techniques using deep learning. More specifically, my goal was to integrate neural networks into a Monte Carlo tree search (MCTS) program written in C++ using TensorFlow Lite.
+ Becoming a competitions master: A silver medal was OK, but even it would have been difficult if I hadn't done the imitation learning.

## Overall architecture

[architecture]

## State transition function (Game program)

During the first month of this competition, I implemented a state transition function (clone of the Lux game program) for my MCTS program in C++. This process was a bit tedious, but important because without it, an efficient MCTS agent could not be created. Moreover, it runs much faster than the original TypeScript program, so it was useful for self-play programs for generating training data for my neural networks.

At this time, I did not implement some specifications such as transfer and pillage in my game program, but not implementing transfer was a big mistake. (Since the order of the unit's actions (move, build city, transfer) was not specified in the specifications, I was not confident in creating a program that behaved exactly the same as the original program.)

## Tree search

I have created a special version of MCTS algorithm for handling simultaneous actions and large action spaces. I will explain this part in detail as it may have been one of the most unique trial of this competition.

[tree-search]

+ In Lux, the number of possible actions to take explodes as the number of units increases, so it is impossible to explore all the actions. Therefore, I programmed it to explore only a few actions sampled from the action generator (I called it ProposalPolicy). The size of the vector of ActionNode is gradually increased according to the number of visits to the StateNode. (Progressive Widening)
+ In a single StateNode visit, each player selects an action from the vector of ActionNode according to the UCB1 algorithm.
+ According to the selected pair of actions, the game state is updated, and the corresponding TransitionNode is selected (or created). If the number of visits of the TransitionNode is less than threshold, the state is evaluated by the value network and/or "rollout".
+ The action value of the selected ActionNode will be recursively updated according to the value obtained by state evaluation.
+ If the number of visits of the TransitionNode is greater than threshold, the next StateNode is expanded and the above process is recursively executed for the node.
+ The tree search is executed in multiple threads, and the "virtual loss" technique encourages parallelism.
+ If the time limit is exceeded or another search is likely to exceed the time limit, the search is stopped and the ActionNode of own team of the root StateNode with the highest number of visits is selected.
+ Time control has also been introduced. If the action value of the selected action fell below 0.42 (this means my agent is in a bit of a disadvantage), I have programmed it to continue tree search for an additional 3 seconds. Of course, it is managed so that the given 60 seconds will not be used up.
+ Another option is to randomly select multiple neural network-based policies and rule-based policies and let the MCTS choose between them. However, in the end of the competition, the neural network-based policy was much stronger, so the weight of the rule-based policy was set much lower.

The drawback of this method is that the exploration space is very small compared to the action space, and the skill of this agent depends mostly on the strength of the ProposalPolicy alone. However, the effect of exploration is certain, it increases the score on LB over 100 points.

## MCTS + rule based agent

My earliest agents were based on combination of the above MCTS and rule-based (but probabilistic) tree and rollout policies. These rule-based policies were relatively simple, combining rule-based task assignment and breadth first search. These agents were in the silver medal range when they were first released, but their rankings dropped significantly due to the public notebooks by @huikang and @shoheiazuma. Therefore, I started to create agents combining MCTS and neural networks as originally planned.

## Policy network

My policy network was created based on the @shoheiazuma's imitation learning notebook. The main differences are as follows,
+ The architecture of the network is a ResNet with depthwise convolution and squeeze-and-excitation blocks. I focused on the tradeoff between computational complexity and accuracy.
+ The "output actions of all units at once" technique was essential to incorporate policy network into MCTS.
+ Probabilistic policy: The output of the softmax function is the probability of each action.
+ Data augmentation (flipUD, flipLR, transposeXY) is performed during training time.
+ Including "do nothing" (move center) increased my agent's score by over 100 points.

[policy-network]

+ The score of my agent using only the policy network is about 1530-1550, but when value network and MCTS are combined with it, the score increases significantly to 1660-1700. It's amazing, isn't it?  (By updating the policy network and the value network, the final score was 1743.1.)
+ After training the policy network, the actions generated by my MCTS agents were further added to the training data as weak labels to increase the robustness of the policy network.
+ The citytile policies are rule-based, just like the original IL notebook. This may have been a weakness of my agents.

## Value network

+ The initial state is generated by self-play using some policy until some turns, then self-play using the policy network is repeated multiple times from the initial state, and the value network is trained using the initial state and the winning rate of team 0 as input and output data.
+ This value network can be viewed as a surrogate model for computationally expensive rollouts using the policy network.
+ The architecture of the value network is a classical pooling-based CNN, and introducing antisymmetry about players improves the generalization performance.
+ By replacing the rule-based rollout with the value network, the score of my agents increased. However, the accuracy of the value network was sometimes clearly worse at the end of the game, so I used the ensemble of the value network and rule-based rollout in the latter half of the game to evaluate the state.

[value-network]

## Comments

+ I was a little surprised to see so few participants using MCTS in this competition. At the beginning of this competition, I thought that all the top players would be MCTS!
+ My agents could have reached a much higher ranking if I had implemented resource transfer and citytile policies properly, but reinforcement learning made it difficult to identify the problem, so when I noticed it, there was almost no time left.

Finally, I would like to thank @stonet2000 for organizing such an interesting competition.
