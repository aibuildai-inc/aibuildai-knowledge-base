# 5th Place Solution

Competition: neurips-2023-machine-unlearning
Rank: #5
Source: https://www.kaggle.com/c/neurips-2023-machine-unlearning/discussion/459148

First of all, thank you to the competition hosts for a most interesting competition. I really enjoyed getting familiarized with the research topic of machine unlearning and this competition. 

My approach is quite simple but effective. I tried to make small incremental changes to some very simple basic ideas which I will share here. This strategy turned out to be quite successful, landing me a 5th place on the private leaderboard and a 6th place on the public one.

The code notebook for my solution can be found here: [Prune/Re-init & Entropy Regularized Fine-Tuning](https://www.kaggle.com/code/sebastianoleszko/prune-re-init-entropy-regularized-fine-tuning)

# Overview
My solution consisted of two primary contributions, 

1. Introducing sparsity through pruning and re-initializing
2. Fine-tuning the model with both regular cross-entropy loss and a kind of regularization on entropy

First, the model was pruned (99% of the parameters) based on an unstructured L1-norm criteria and then re-initialized to random weights. The idea of this step was simply to both unlearn the information from the forget set, and to add sufficient noise to the model to "move" far enough away from the initial model that we can properly imitate the distribution of fully retrained models. Then, the model was fine-tuned on the retain set but with a modification to the loss,

where L_FT is the cross entropy loss for fine-tuning on the retain set and L_RegEntropy is the MSE of entropy between the original network’s (trained on all data) predictions and the fine-tuned networks predictions. The idea is that this acts as a sort of regularization on the fine-tuning, making sure that the entropy is similar to our starting point which is also the best estimate of entropy for the retrained models. 

With these two steps, my thinking was that the model should have (1) forgotten the information of the forget set, (2) moved far away from the fine-tuning starting point which was the full data model, and (3) been fine-tuned back to high performance and trained in such a way that the resulting predictions are similar to those of a model trained with similar parameters as the original training.

# Important ideas
## References and inspiration
As I developed the general idea of my solution, I found two articles that either inspired me or to some extent confirmed my choice of methods. Although I didn't use any of the suggested methods directly, these papers gave me valuable insights into SOTA approaches and why they might succeed. I built on those results and found simple steps that led to improvements in the competition.
- [Model Sparsity Can Simplify Machine Unlearning](https://arxiv.org/pdf/2304.04934.pdf)
- [Towards Unbounded Machine Unlearning](https://arxiv.org/pdf/2302.09880.pdf)

## Hyperparameters
As I started to experiment with different ideas in the competition, I quickly found that the most important thing for me would be to properly tune the hyperparameters of the model. Although I understand the risk of overfitting to the public leaderboard when tuning against submitted results, I think that the value of doing so outweighed the risk. I think more than half of my submissions were spent tuning well performing methods' hyperparameters in small steps. This could often lead to jumps in the public leaderboard of 0.005+ at a time and quickly moved me up the leaderboard. I found most success in tuning the learning rate/number of epochs along with adding a very small amount of class weights to the cross entropy loss term w^(0.1). I never managed to achieve good results with the maximum number of epochs that I could train with the time limitations, however I expect that this would have helped if I managed to tune the rest of the parameters optimally for this (scheduler, learning rate, class weights etc.).

My reasoning of why hyperparameters were especially important in this competition was due to the lack of precise information about how the original model was trained. Since I believe that this information is required to make an efficient unlearning algorithm under the proposed evaluation metric, the next best thing was to systematically guess parameters until the score went up. I would have liked to have more information about the training procedure to avoid this guesswork for future unlearning competitions.

## What I would have liked to try
- Other pruning criteria such as removing parameters with high importance for predicting forget set images but not others
- Maximizing the allowed time to run the algorithm (more epochs)
- Further explore if the MSE loss was better than something like a KL divergence or cross entropy. This is something that I changed a couple of times without improvement on the public leaderboard. This loss term was possibly one of the most important pieces of my final solution.

# Conclusion
This competition had me looking for new exciting solutions from start to finish but I ended up with quite a simple but efficient method. I tried a lot of other ideas but I either didn't have the time to properly tune them, thus couldn't know if they actually improved my unlearning algorithm, or they didn't work for me. I do however believe that the key was in somehow performing the three steps that i outlined in the summary:
1. Forget the data (prune)
2. Move the model randomly away from the fixed fully trained model (randomly re-initialize parameters)
3. Regain performance while not moving too far from the original model (Regularized Fine-tune)

I look forward to learning more about unlearning in the future and to follow the work of my fellow Kagglers and researchers. Thank you again to the competition hosts and to everyone who shared their solutions and insights, I've really enjoyed reading up on how you all have approached this competition!

###  Final scores
Public LB: 0.0932121752
Private LB: 0.0863052371
