# 6th place solution

Competition: playground-series-s4e2
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s4e2/discussion/480795

first thanks to @divyam6969 for your helpful notebook. I edited it to make it a massive mess so not going to post it. 

The main part of the model was from the notebook (ensemble of the classic tree algorithms ect). I then added a simple neural network into it where I optimised the parameters with optuna. I am very new to data science, but my thinking was that a neural network would learn differently to the tree based algorithms, so it would be good to keep it in for stability purposes, even if it didn't improve the cv much. Also the data was generated from a deep learning model, so it would probably be a good idea to incorporate one.  

The second thing i changed was optimising the weights of the ensemble with a simple grid search(i'm not sure if this is good practise but went with it because it increased my cv).

I had a lot of models with higher LB scores but I followed @thomasmeiner advice and treated it as a fold in cv, so submitted models with high cv, but cv that didn't deviate when submitted to public LB.

The last and most important thing was luck :)
