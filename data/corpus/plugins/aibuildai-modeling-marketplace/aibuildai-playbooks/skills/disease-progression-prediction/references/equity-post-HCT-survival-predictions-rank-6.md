# 6th Place Solution

Competition: equity-post-HCT-survival-predictions
Rank: #6
Source: https://www.kaggle.com/c/equity-post-HCT-survival-predictions/discussion/566686

Hey everyone,


I've made public the notebook that earned me 6th place:
https://www.kaggle.com/code/myprofileurl/6th-place-two-step-model

Here is the summary of what it does

### 1. Gradient Boosting Models  
- Use gradient boosting models to predict:
  - **p(efs == 1)**: the probability that the event occurs.
  - **E[efs_time | efs == 1]**: the expected survival time conditioned on the event occurring.

### 2. Neural Network  
- Train a TabM neural network that optimizes a smoothed version of the evaluation metric (using a sigmoid instead of the indicator function).

### 3. Combine the Predictions  
- Run 20-fold cross-validation with 5 random seeds for all models.
- Train a one-layer neural network (with no non-linearities) on out-of-fold (OOF) data, with the same loss function as the TabM model.  
- **Inputs to the NN:**
  - **Risk score from the TabM model**
  - **2nd-order polynomial combinations** of **p(efs == 1)** and **E[efs_time | efs == 1]**

I can publish code for individual parts of the solution as well if anyone is interested.
