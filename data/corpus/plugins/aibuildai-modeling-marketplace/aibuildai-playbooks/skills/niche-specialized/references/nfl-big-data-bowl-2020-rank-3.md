# Public 5th place solution overview

Competition: nfl-big-data-bowl-2020
Rank: #3
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119357

I want to give a big shout out to the National Football League and to Kaggle for hosting such an interesting competition! Personally, what I liked most about this competition was that it challenged you to build a custom model which reflects the dynamics of the game.

The competition instantly reminded us of the [Molecular Properties competition](https://www.kaggle.com/c/champs-scalar-coupling), where one was asked to predict scalar coupling constants based on the spatial configuration of atoms. With that, our idea was to incorporate ideas from there into our models. We had the 2nd place solution of Quantum Uncertainty in mind: 
&gt; This immediately triggered the idea of using transformer layers (encoders) stacked taking as an input x,y,z (normalized but otherwise as-is), and atom type and j-coupling type embeddings; just concatenated… nothing fancy.

We therefore did minimal feature engineering and used mainly the data preprocessing steps which can be found in the public kernels. All plays were transformed such that the play direction is towards the right. Ordinal features were transformed using StandardScaler.

Our model consists of three distinct blocks, a **player-player interaction block**, a **player block** and a **global game block**. 

**Player-player interaction block**: 
Input is a tensor `X_1` of shape `(batch_size, num_players, num_players, num_features)`, where `X_1[:, i, j]` contains information about player i and player j - their relative distance, velocity and acceleration (distance also extrapolated into the future), their relative distances/velocities/accelerations in x and y direction respectively, as well as their team - offense/defense/rusher. 
`X_1[:, i]` is then fed into one of three possible Transfomer-type blocks. That is, we have one Mini-Transformer (just as in BERT with 2-3 Blocks) for offense players, one for defense players and one for the rusher. After these blocks, we concatenate the outputs together (same shape as `X_1`) and take the mean over the tensor dimension 1. With that, we have a tensor `X_1’` of shape `(batch_size, num_players, num_features’)`

This output is then concatenated along the last axis to the **Player input tensor X_2**, which has shape `(batch_size, num_players, num_features’’)`. It contains features such as position of each player, distance to rusher, velocity in x and y direction, etc. The concatenated tensor is processed with another Mini-Transformer (usually 2 Blocks). After that, we keep only the first dimension of the output `X_2[:, 0]`, where index 0 is the position of the rusher, and concatenate it with the **play input**. 

The play input contains information such as distance to yardline, etc. The concatenated tensor is then processed with the prediction head (just 2-5 linear layers) to predict the cumulative probability distribution of shape `(batch_size , 199)`.

Before the sigmoid output, we add a mask such that “inaccessible” yards (as determined by the current yardline) are predicted correctly (the mask is either -100, 0 or 100, such that `sigmoid(x+ mask)` will be 0 or 1 for regions for which we know the actual label). This gave us a boost of about ~ 0.0003 and stabilized the training of the model.


Model development was performed using time-based 5fold splits. As the validation loss varies a lot throughout the folds (from 0.010x to 0.013x), we added additional metrics which computes the MSE loss on long runs (which we defined as runs with Yards&gt;=15) and short runs (runs with Yards&lt;15). The long run loss is about ~10 times larger (0.07-0.08) than the short run loss (0.007-0.01, depending on whether some rushers run negative yards). The difference in validation loss can be thus explained by the percentage of long runs which varies among the folds. We used MSE loss, a batch size of 32, 5% swap dropout for certain features after epoch 4, and learning rate decay with a start learning rate of 5e-4. The transformer hidden size of our models ranges between 32-48, with either 1 or 2 attention heads.
We did not scrape the test data for model development.

There were numerous tweaks on how to improve the model’s performance, e.g. it was important to leave the distance features in the player-player interaction block unscaled (no preprocessing). Also, the splitting the interaction into different parts (offense/defense/rusher) improved our score. We implemented a TimeDistributed layer which processes X_1 in parallel. This reduced training time by about 40% compared to a python loop over the players. We also applied some data cleaning on the 2017 games.
For our final submission, we used 3 and 4 different Model architectures (the three model notebook is conservative regarding the run time). Each model is trained up to 12 Epochs and the two best epochs per model are kept for prediction. 

What did not work: A lot! We tried to incorporate the last play’s features into the model, used aggregated information about the rusher and added several features which we thought were meaningful input. In all cases the model’s performance was about the same. I guess this is due to face that the geometrical information is the most important factor, whereas other features’ impact tends to fade out if it they are not crucial. We also experimented with different loss functions, giving higher loss to either long or short runs without any decisive advantage.
