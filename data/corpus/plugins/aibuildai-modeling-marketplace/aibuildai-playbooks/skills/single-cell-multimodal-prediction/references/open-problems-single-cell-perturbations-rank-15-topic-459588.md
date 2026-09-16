# Public 7th and Private 15th solution (Nothing but just multiplied a factor of 1.2)

Competition: open-problems-single-cell-perturbations
Rank: #15
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/459588

First many thanks go to those who created 0.574 and 0.577 public notebook.

While I started this competition 2 months ago, I found the public LB is very variant, it’s very different from CV. Ensemble some worse LB results like everyone (or most people) found that ensemble results with 0.702 LB will boost the result. So I thought this would be a shake competition. Hence, I didn’t spend too much time on how to build a more diversity or robust model (I didn’t think I could), but I focused on some LB probing or tricks. For instance, multiplying a factor to any of my results (the best one is 1.2 on public LB), I could boost my results. This made me more confident that there would be some big shake up, but I also believed some of the gold medal teams will be quite stable, they will stay there.


I used public 0.574 and 0.577 results + some of my own models (Public LB 0.578).

For my own models:
•	Conv1D NN
•	LSTM
•	MLP
•	LGBM

Features:
•	Standard Scaler train label columns for NN models
•	One-hot encoded cell_type,sm_name 
•	Split SMILES to character and use TFIDF to get embedding.


Final results:
Step1
•	sub_pub[:128] = 0.55*Public 0.574[:128] + 0.45*public 0.577[:128]
•	sub_pub [128:] = 0.6*Public 0.574[128:] + 0.4*public 0.577[128:]
•	then postprocess it using what’s done in https://www.kaggle.com/code/jeffreylihkust/op2-eda-lb 

Step2
•	final_sub = 1.2*(0.95*sub_pub + 0.05*my_0578)

About the factor:
I tried factors of 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, but 1.2 gives best public LB.

A bit pity, I have a few results in the gold area, but their public LBs is 0.02 worse.
