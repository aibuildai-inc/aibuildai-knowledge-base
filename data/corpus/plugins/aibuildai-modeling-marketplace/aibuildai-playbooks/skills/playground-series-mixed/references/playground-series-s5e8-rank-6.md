# 6th Place Solution  ->  OOF Stacking with LGBM

Competition: playground-series-s5e8
Rank: #6
Source: https://www.kaggle.com/c/playground-series-s5e8/writeups/6th-place-solution-oof-stacking-with-lgbm

First of all i want to apologize for the relatively undetailed writeup of me, i worked really unorganized and really didn't think i would make it to the 6th Place, but if you have any quetions  feel free to Post them, i will answer them.
My Solution is every similar to the solutions committed  in the Top 10, but to be honest i had a very limited Number of Models and OOFs compared to what i have see so far. 
My submission were 7 Models Stacked with a optuna hyperparameter optimized LGBMClassifer --> CV: 0.97736 / Private LB: 0.97766
When i add the original Features to the Meta Model as input, it gave me a significant boost in both CV and LB -> CV: 0.97739 / Private LB: 0.97775
This one was my Final Submission which gave me the 6th Place.

The 7 Models i used as Input for the Meta Model were:
The CNN and the TabR from @omidbaghchehsaraei, absolute thanks to him, i learned a lot via his Notebooks and suggest you to look into them if you haven't.
I also think these 2 Model gave me a boost of diversity in my Models and there for improved my Score significantly.

The other Models i had were:
LGBM tuned with Optuna
XGBoost tuned with Optuna
2 Catboost tuned with Optuna
and a NN.

For Feature Engineering i mostly used (not exclusivly) the Target Encoding with the original dataset presented by @cdeotte in his XGBoost Notebook.

Actually for the KFold Splits, i always used different splits and random_states for the OOFs depending how long a model would train. 
I know that i should have done the OOFs with the same split and random_state, but i found out to late and didnt have any compute left on Kaggle :P.

Congrats to @optimistix , @mahoganybuttstrings and @bestwater for the 1st, 2nd and 3rd Places
