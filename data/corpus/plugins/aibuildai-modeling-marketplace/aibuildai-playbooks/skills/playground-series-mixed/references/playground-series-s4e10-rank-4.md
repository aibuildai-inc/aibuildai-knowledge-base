# Rank 4 approach - thoughtful model choices and effective ensembles

Competition: playground-series-s4e10
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s4e10/discussion/543672

Hello all,

Thanks to Kaggle for a good classifier episode in the Playground series! I also wish to extend sincere thanks to all the participants in the competition and the forum contributors for their wonderful contributions though the month. I also extend sincere thanks to the community for receiving my artefacts so well in the challenge! Please find below my approach for the competition and the associated write-up - 

## Approach overview 

My overall approach for the competition was simpler than the recent past editions of the playground series, as I relied on a lesser number of features and more of mindful blending and stacking in this episode. I was of the opinion that a smaller data size could increase one's chances to overfit and hence, retained a simple approach with boosted tree and NN model options **without blind blending and with careful consideration for data leakage and cv-scheme analysis**

I used a simple stratified 10-fold CV scheme with random state = 42 in this challenge and retained this for all the models uniformly. 

My overall approach for the chosen submission can be illustrated as below- 



<br>

I managed my time in this competition as below-
1. Consider a single model - say Catboost. Work on the model with multiple features and parameters till I am satisfied with the signal extraction using the model
2. Proceed to the next model, say LGBM. Repeat the process on various feature sets till I am confident with the results/ run out of time 
3. Repeat the process with different algorithms (single models) till the final week
4. Consider various blending and stacking options on the best CV results across single models 
5. Make a final submission choice based on cv-scores

I wish to extend special thanks to the kernel [here](https://www.kaggle.com/code/mikhailnaumov/loan-approval-ensemble-nn-xgb-lgbm-cat) - this helped me greatly in my final push on the leaderboard. 
My other submission was my own work entirely excluding the public kernel artefacts. I believed in my contributions in the competition and hoped that my work would prevail and am happy my belief indeed prevailed! CV- LB relations for the 2 submissions are as below-

| Submission details  |  CV <br>10-fold Stratified K-fold| Public Leaderboard | Private Leaderboard | 
| --- | --- | -------------- | ------------------- | 
| My independent work + public work - stacked with torch NN  | 0.97002 | 0.97393 | 0.96902 | 
| My independent work without public work   | 0.969954 | 0.97353 | 0.96899 | 

<br>

## Feature sets and Feature engineering

I relied on 5 feature sets in a feature store, akin to my [TPS- July 2024](https://www.kaggle.com/competitions/playground-series-s4e7/discussion/523404) process. I designed a gamut of features, featuring a lot of brute-force driven secondary features and varying model parameters, but nothing prevailed more than a **simple catboost model with all string values**. I presume this dataset was designed in a manner that perhaps was not amenable to secondary features. **I used the original data in all my feature sets**.

I ended up using 3 feature sets out of 5 sets based on the CV scores and choose diverse models to my best capabilities to stack up with a neural network. I discarded my feature sets with 50+ features as the CV results were not encouraging at all. Upon peeking into the private scores across my single models, I am happy I rejected the said features and relied on simplicity. My chosen feature sets included between 15-25 features, all created with simple operations on the existing features **without data leakage**.

## What did not work
1. Autogluon models
2. Xgboost - I used the XgBoosts posited in the public work but my private XgBoost models always failed on the LB
3. Linear models like logistic regression
4. Random forest
5. KNN 

## My modus operandi and key take-aways

I presume a lot of beginners will read this post and hence, wish to add a separate section on my take on submitting well in this series- 
1. Take your time with your first real and meaningful submission - build a pipeline first and ensure it works well. This is highly important for a smooth experiment process through the month
2. You may consider my [public artefacts](https://www.kaggle.com/code/ravi20076/playgrounds4e10-baseline-v1) as a base model - I have highly simplified the process to remove clunky elements and have retained and developed a modular structure for convenience. This helps me manage my models effectively and easily
3. Always save your fitted model, OOF score and test set scores with a proper naming convention. This will help you immensely when you blend and fuse models at a later date
4. CV-scheme is one's open secret to success - **ignore anything that does not have a proper cv-backup and justification**
5. Don't rely on public materials without testing them for compatibility with your process - this is imperative for you to generalize your models well. I encourage one to use the ideas shared in public kernels in **your independent work**
6. Team up well and learn together - we are here first to learn and then for results 
7. A well organized GitHub repo is a great addition - please consider making a repo for your work and place your code and artefacts there. It will immensely help you across competitions
8. **Dwell on your mistakes in one episode and learn from them first before moving on - this is a long run learning and is likely to improve you in future episodes**
9. **NEVER BLINDLY BLEND PUBLIC KERNELS** - this is almost certain to fail 

## Concluding thoughts
I wish to extend sincere wishes to one and all for Diwali and hope the festival of lights and joy brings in a lot of happiness for all of you on Kaggle and in other walks of life as well!
See you all in the next episode of the series and all the best!

Regards,
Ravi Ramakrishnan
