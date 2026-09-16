# Wow (and our solution)

Competition: icr-identify-age-related-conditions
Rank: #2
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/430860

To put it mildly, I'm in complete shock, I woke up today with my teammate congrats and I got intrigued as to what place we got? I burst out laughing when I saw it was 2nd place, and the funniest thing is my solution is just CV - no probing, no nothing

here's my solution:

https://www.kaggle.com/code/opamusora/main-notebook

What we did:

- just like many people, we used time and max(time)+1 for test
- removed rows where time is None, I noticed a weird cluster that was far away from all other data when playing with umap and it were rows with absent time
- filling Nan values with -100, probably doesn't matter if its median or low numbers.
- reducing dimensions with umap and then labeling clusters with kmeans, it didn't bring a lot of score, but it's there
- did feature permutation manually, dropped any cols that made score even slightly worse
- for a model I used Catboost, xgb with parameters from some public notebook and tabpfn. LGBM didn't seem to work for me as it always dropped my CV
- then we just averaged our predictions for test and that's it

Also, I want to mention that we wanted to try edit tabpfn to get embeddings and we had an idea to try fine tune tabpfn, but it didn't work out.

I also tried optuna for optimizing my models, but it didn't work out

Stacking didn't help my score much as well

edit: My last 2 sumbissions on this competitions are the highest scoring ones 😅
