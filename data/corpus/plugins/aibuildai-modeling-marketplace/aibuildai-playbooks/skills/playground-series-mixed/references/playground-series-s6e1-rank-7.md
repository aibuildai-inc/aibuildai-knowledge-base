# Private #7 | Diversity Ensemble & CV Trust

Competition: playground-series-s6e1
Rank: #7
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/private-7-diversity-ensemble-and-cv-trust

My final solution is a Ridge stacking ensemble built on 31 level-1 models.
The base models come from different model families and use different data treatments to encourage diversity.
Although the overall feature engineering is relatively simple, I introduced variations across models, including:

slightly different feature engineering pipelines,
log1p transformations,
sqrt transformations,
residual modeling.

All level-1 models were trained using 5-fold cross-validation with a fixed random state of 42.
Out-of-fold predictions from these folds were used as inputs for the Ridge stacking model.
These variations were designed to reduce correlation between base models rather than to maximize the performance of any single model.
The final predictions are obtained by stacking all level-1 model outputs using Ridge regression




I am especially grateful to everyone who shared their models, techniques, and practical tips. These contributions saved me a lot of time and helped me avoid many pitfalls. @siukeitin @cdeotte @masayakanagawa @yekenot @omidbaghchehsaraei 

I did not focus heavily on feature engineering. Instead, my main effort was on model diversity.
I also did not perform extensive hyperparameter tuning. Since I did not go deep into feature engineering, the feature sets across different models were largely similar.
To introduce diversity, I applied different treatments to different models, such as log1p transformations, residual modeling, and slightly varied feature engineering strategies, along with a few other techniques.
In the later stage of the competition, my models started to stall on the leaderboard and even decline. At that time, I over-trusted the public LB, which caused me to pause training more diverse models. I was worried that adding more diverse models would introduce noise into the stacking process.
Instead, I focused on training very similar models, such as changing a few parameters, using different seeds, averaging, and applying different stacking strategies. This actually introduced more confusion rather than improvement. As a result, CV barely improved, while the public LB kept dropping until the end of the competition.

Fortunately, in the final submission, I chose:

the model with the best CV (trained 5–6 days earlier, before the LB drop),
together with the best LB-performing model,

and finally selected the submission with the strongest CV overall.
This decision helped me finish Top 10 on the Private LB 🙂

Due to a lack of self-management, combined with a continuous decline in the public LB during the later stage of the competition, my workflow became increasingly chaotic, which led to several poor decisions.
I abandoned multiple models that showed moderate but consistent improvements in CV, simply because their public LB scores dropped. Many of these models were never added back, revalidated, or properly analyzed until the end of the competition.
Due to these issues, many ideas and techniques I had in mind were never fully implemented or properly validated.
Some approaches were only partially tested, while others were abandoned before they could be integrated into the stacking framework or carefully evaluated through CV.
This competition taught me a lot.
At the same time, it clearly exposed my weaknesses in process management and decision discipline, especially my tendency to over-trust the public leaderboard.
That said, despite the mistakes and inefficiencies, the overall outcome was still positive.
The experience, lessons learned, and final result ultimately outweighed the drawbacks.
