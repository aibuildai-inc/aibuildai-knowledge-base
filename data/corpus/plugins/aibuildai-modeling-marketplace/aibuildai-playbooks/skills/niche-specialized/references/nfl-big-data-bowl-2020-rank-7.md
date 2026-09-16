# Private 7th(Public 12th) solution&code with keras GCN library

Competition: nfl-big-data-bowl-2020
Rank: #7
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/124999

I am very grateful to competition organizers and Kaggle for holding this great competition. There are many different approaches like CNN, Transformer, GCN, MLP, Boosted Trees, and many feature engineering, and there was much to learn :)

My codes are available from : https://github.com/vinmuk/NFL-predict-yards
Useful keras GCN library : https://github.com/danielegrattarola/spektral

# Overview
- Data fix
  - No data modification.
  - With some features, there was a difference in distribution, however, in my case, fix them by adjusting mean and standard deviation makes score worse.

- Model
  - 2-layer GCN with multi output.
  - Attention sum pooling layer (attention is computed by dot products of trainable weights and features of each player) follows each GCN layer 
  - GCN layer is my original layer which combine Graph attention networks (GAT) and GraphConvSkip layer.
  

- Optimizer
  - Adam(lr=1e-3)

- Loss function
  - Binary crossentropy and mae for the last layer
  - Binary crossentropy loss with divided output into 50 sections are located immediately after each GCN layer. This hastens the time to convergence.

- Data Augment
  - Flip y axis
  - Add data after a few seconds doesn't work
  - TTA doesn't work

- Feature engineering
    - 115 features for each player
    - Separate player feature and play feature made score worse
    -  Below features after 0~2 seconds significantly improved score.
       -  the number of opponent players around each player.
       -  whether to collide with rusher when traveling a certain distance
       -  distance from rusher

- Post process
  - Padding by 0 or 1 for parts that cannot be reached from the current position.
