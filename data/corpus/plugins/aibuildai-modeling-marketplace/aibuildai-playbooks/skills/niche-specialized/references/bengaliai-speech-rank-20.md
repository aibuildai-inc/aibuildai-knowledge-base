# 20th Rank Solution

Competition: bengaliai-speech
Rank: #20
Source: https://www.kaggle.com/c/bengaliai-speech/discussion/448066

Our Final Solution is a single Wave2Vec model along with a N-Gram Model

# Step 1:
Trained a model for 5 epochs on 
1. OpenSLR37
2. OpenSLR53
3. Fleurs
4. Common voice

Validation:
Kaggle Validation data

# Step 2:
Estimate the WER for the Kaggle training data with the model from Step 1.
Remove samples which have WER greater than 0.5
Resulting cleaned version of Training data had 50-60% of overall Training data

# Step 3:
Trained a model for 20 epochs on 
1. OpenSLR37
2. OpenSLR53
3. Fleurs
4. Common voice
5.  Cleaned version of Training data

# Step 4:
Build a N-Gram model with Common Voice dataset

Our best Wave2Vec model                                       0.407 in LB
Our best Wave2Vec model + N-Gram Model          0.396 in Public LB

**Things that worked:**
1. Caching of the datasets reduced the overall training time. But it didnt allow us to add any augmentations on the fly
2. Having more warmup steps ensured that the model doesn't throw NaN. We used warmup of 0.25
3. MSD as Final layer instead of Linear layer

**Things that didnt work:**
1. We tried adding few more finetuning steps. But mostly the model started to overfit

We will add few more points about our pipeline in next few hours
