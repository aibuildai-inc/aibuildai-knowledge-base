# 4th Place Solution

Competition: predict-student-performance-from-game-play
Rank: #4
Source: https://www.kaggle.com/c/predict-student-performance-from-game-play/discussion/420349

# Acknowledgement
I'd like to thank the hosts for providing a very interesting and difficult project to work on for the past months. I am also grateful for all the public sharing on Kaggle, this has been an insane learning experience for me. Without all the public notebooks, discussion posts and old competition solutions available i would have had no chance in this competition.

# Overview
- Used most of the raw data for training, while validating only on the kaggle data.
- Ensemble of Transformer, XGBoost and Catboost, with 3 seeds and 5 folds each. 
- Used a generic set of features based on time, index and screen_coor differences.
- Linear regression as a meta model.
- Thresholds have a big impact on LB score

# Data
I used most of the raw data for training, including sessions that only completed level group 0-4 and 5-12. About ~38000 whole sessions and ~58000 sessions in total. Using the raw data increased CV by over 0.001. I validated only on the kaggle data. 

My initial data preprocessing is simply sorting by level group and index, same as what happens during inference. Also, my experiments indicated no benefit from using the hover durations, so after sorting i dropped the hover rows and re-indexed each session from 0 to len(session).

# Transformer
I spent much of my time experimenting with transformers, which resulted in a light weight model that achieved 0.698 on the public and private LB, and 0.702 CV. 


```
class NN(nn.Module):
    def __init__(self, num_cont_cols, embed_dim, num_layers, num_heads, max_seq_len):
        super(NN, self).__init__()
        self.emb_cont = nn.Sequential(
            nn.Linear(num_cont_cols, embed_dim//2),
            nn.LayerNorm(embed_dim//2)
        )
        self.emb_cats = nn.Sequential(
            nn.Embedding(max_seq_len + 1, embed_dim//2),
            nn.LayerNorm(embed_dim//2)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=0.1,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.clf_heads = nn.ModuleList([
            nn.Linear(embed_dim, out_dim) for out_dim in [3, 10, 5]
        ])

    def forward(self, x, grp):
        emb_conts = self.emb_cont(x[:, :, :-1])
        emb_cats = self.emb_cats(x[:, :, -1].type(torch.int32))
        x = torch.cat([emb_conts, emb_cats], dim=2)
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.clf_heads[["0-4", "5-12", "13-22"].index(grp)](x)
        return x.unsqueeze(2)
```

- embed_dim: 64
- num_layers: 1
- num_heads: 8
- max_seq_len: 452 (explained below), though the sequences are cropped to 256
- I used the same single model for all questions

I found the data easy to overfit with transformers, so in an attempt to improve the signal to noise ratio i did the following:
1. Identify different points in the game by string concatenating event_name, level, name, page, fqid, room_fqid, text_fqid, in the dataframe.
2. Some of these occur more than once in a session. Treat these as different points by enumerating them and adding the enumeration to their names.
3. Filter out the rows with points that is present in over 0.999 of the sessions. This makes each session maximum 452 steps long. 
4. Create 6 feature columns:
time difference, index difference, distance (cumulative distance moved, calculated from screen_coor's) difference, room_coor_x, room_coor_y and the categorical point column embedded.

# XGBoost
This is my strongest single model with public LB 0.701, private 0.702, and 0.7029 CV. What stands out is that i flattened 5 of the transformer input columns (excluding the categorical column), and used all those values as individual features.

The other features are mainly stats that can be found in public kernels, like mean and max time diff over the categoricals. The stats were calculated before applying the transformer input filtering.

From early on I trained one model for each level group, inputing the question number as a feature. I found that CV increased by around 0.0002 compared to using a model for each question. This could be randomness, but i went with it since i thought 3 models instead of 18 would make my life easier during experimentation. Similar reasoning behind using one model for the transformer. 

# Catboost
Essentially looks the same as XGBoost. CV 0.7022.

# Ensemble
I trained a linear regression meta model for each question, using the above models output probabilites as input, to produce the final predictions. I included probabilities of past questions and some future ones! For example the regression model trained to predict question 2 took probabilities on question 1-3 as input, to predict question 7 I used probabilities on question 1-13, and for question 16 i used probabilities on question 1-18. I took 3 seeds average before linear regression input to make it more robust. 

This finally results in public LB 0.702, private LB 0.703, CV 0.7044. 

# On threshold and submission selection
I tried to trust CV as much as possible, but the consistent gap between my CV and LB was suspicius until the last few days. Then I realized one reason could be my selected threshold was suboptimal on the test data.. I made some submissions with my highest CV solution, only changing the threshold, and noticed it was indeed suboptimal and caused more variation in LB score than most of my latest experiments. So in the end I selected 3 of the same solution, with different thresholds: 0.60 (best on LB), 0.62 (best during CV) and 0.64. Turned out 0.61 would have resulted in 0.704 private, but no regrets ;)

Thank you for reading!

# Code
[Training code](https://github.com/joelerikanders/pspgp/tree/main)
[Submission notebook](https://www.kaggle.com/erijoel/4th-place-submission)
