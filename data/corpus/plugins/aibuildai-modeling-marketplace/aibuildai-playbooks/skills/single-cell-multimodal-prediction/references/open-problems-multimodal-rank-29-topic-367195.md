# private 29th place (public 4th)solution

Competition: open-problems-multimodal
Rank: #29
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/367195

Thanks to the organizers and to everyone who share their ideas in public notebooks and discussion. I also needed to gain experience in analyzing single cell data and this competition help me gain a lot of knowledge and will become a precise experience.

I'm a postgraduate student and it's my first time to join the Kaggle competition, so if my organization in this nootbook was not clear and you want to know more other information, you can comment below or send me a private message. As a result of I didn't control the time very well, so the final plans were not been finished and submitted. My focus was on the feature engineering, some methods such as DCA, Magic, TruncatedSVD, FA, LDA and so on can make a positive effect in the final result. About the cross validation part, I divided the data by batch, but I don't think it's a good method for the private score. About the ensemble part, I selected the models including NN, Lightgbm, Catboost, Xgboost and Kernel ridge(For citeseq is ridge).

Because the feature engineer's parts are not organized well, now I just to share the NN model structure which gained the best score among single models, though the final TruncaredSVD's parameters were changed a little.[https://www.kaggle.com/songqizhou/private-39th-public-4th-s-basical-single-model-nn](https://www.kaggle.com/songqizhou/private-39th-public-4th-s-basical-single-model-nn).
