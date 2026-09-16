# 3rd Place Solution

Competition: ncaaw-march-mania-2021
Rank: #3
Source: https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/232903

Sorry this is a bit late! First I’d like to say thanks to the Kaggle team. I’ve really enjoyed participating in these NCAA competitions over the years and it’s been great--and lucky!--to finish third this year. I’d like to dedicate the gold medal to my submission's namesake [E. T. Jaynes](https://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes) whose book *Probability Theory: The Logic of Science* radicalized my Bayesianity.

The general approach is to train multiple independent models and ensemble them together. The only ‘gambling’ I did was the standard 0/1 strategy on the first round game I had predicted closest to 50/50. All other games are identical in both submissions. The only data used is box score type data provided by Kaggle and mostly only the game scores at that. I also ran the exact same strategy for the mens tournament--just missed the top 100--but the parameters were optimized to different values.

**MODELS**

I used four different models this year. The first two are models I developed which are similar to each other. The second two might look familiar.

**Gaussian Latent Variable Model**

This is a Bayesian generative model on point spreads. The only data it uses is the point spread in each game and who was the home team. Originally the model assumed that each team had a latent variable with a normal but extremely wide prior corresponding to how strong they are. Given the latent parameters, the point spread sjk in a game between teams j and k is modeled as the normal random variable:

$$S_{jk} | x_j, x_k \sim N(x_j  - x_k - homeadv, \sigma^2)$$

Where the home advantage and variance are hyperparameters that get optimized. So then given a set of game results we can compute the posterior distribution on the team strengths. Since the prior is conjugate we can compute the posterior exactly by solving an Ax = b matrix equation. Probably this is equivalent at least in spirit to some sort of multilevel linear model with team level effects. Roughly speaking the model expects the point spread in a game between two teams to be equal to the difference of their strength latent variables on average.

At some point over the years I realized the quadratic-ness of the normal distribution would make the model over-confident in cases where there was a gap in team strength--e.g. 2 seed vs 15 seed. A lot of this is made worse by games in the training data with extreme point spreads--like an 80 point UConn win. To try and fix this I ‘preprocess’ the input spread data to A) adjust the for home advantage--instead of doing it in the model like above--and B) scale the score so it can’t be too large. This is the function I use:

$$\tilde{s}_{jk} = $$

$$scoremax *  \tanh ( \frac{s_{jk} - homeadv}{scoremax} )$$

--apologies for the messy latex but this line wouldn't render correctly--with scoremax being a hyperparameter. Then these ‘adjusted’ spreads are fed into a similar generative model as before:

$$\tilde{S}_{jk} |x_j, x_k \sim N(x_j - x_k, \sigma^2)$$
 
To get win probabilities we can just compute $$P(\tilde{S}_{jk} > 0 | data)$$ using the posterior distribution on the latent variables.

**Gaussian Latent Variable Model, But Use a Logistic Distribution at the End**

This model is nearly identical to the previous one: doing the same spread adjusting and the same posterior calculation to get the latent variables--though some of the hyperparameters might get set to different values. However, as another way to address the issue of being overconfident when there is a gap between two teams’ strengths I tried something different. Basically, I just assume the probability of team j beating k is a logistic function of the difference of latent variables xj - xk. This tends to be much kinder when  xj - xk is further from zero.

**OneShiningMGF**

This model is basically a rip-off of the model that @statsinthewild and @statsbymichaellopez described in [this paper](https://statsbylopez.files.wordpress.com/2013/08/jqas-2014-0058.pdf). Notably, I did not use the betting odds part of the model, only the KenPom-style possession based stat logistic regression model. Also, I opted to use the 'g' parameterization from Table 2 instead of 'f' like they used since 'g' has fewer coefficients to fit and had similar performance.

Another difference is that I home-rolled my own possession based statistics using the box score data instead of sourcing directly from the KenPom site. It's not an incredibly thorough explanation, but [this page](https://kenpom.com/blog/national-efficiency/) describes how you can recover the adjusted offensive and defensive efficiency ratings using only box scores by what amounts to solving an Ax = b matrix equation. I think this might not be the most current method for the KenPom ratings, but it is what I used. A key advantage of doing it this way is that you can get the ratings given an arbitrary set of game results, so respecting the arrow of time is easier. 

I’ll be sure to give a more thorough writeup of the way I compute these possession based stats when the competition launches next year. 

**Raddar**

How could anyone possibly do well in a Kaggle competition without xgboosting? This model is also ripped off from a past top performer: @raddar . I generally used the same approach to fitting as described in [this notebook](https://www.kaggle.com/raddar/paris-madness) he shared, but fed a different set of features into it. This was mostly since I didn’t have a lot of time to implement everything he originally used and test additional features. So I used:

- KenPom-esque Adjusted Offensive Efficiency--also used in previous model
- KenPom-esque Adjusted Defensive Efficiency--also used in previous model
- Average Point Spread--adjusted for home advantage
- Average Point Spread vs top 50 ranked opponents--adjusted for home advantage
- Win percentage in last 14 days 
- Win percentage in games that were decided by 3 or fewer points or went to OT
- Home team

**ENSEMBLE**

First I took all of the predictions for all of the models and transform them from the unit interval to the real numbers by taking the logit. Then I did a positive coefficient logistic regression with each feature being a logit transformed model. I found that this step would too often put all of the ensemble weight on a single models, so I A) imposed a minimum weight on each coefficient of 1/6 and B) add an extra penalty proportional to minus the L2 norm of the normalized coefficients--i.e. the penalty is least if all the coefficients are the same. I didn’t have a lot of time to test the ensembling method, but the minimum weight and the penalty gave weights that felt right to me: not too heavy on a single model, but still weighed the better models a bit more.

**METHODS**

For each season I defined a set of ‘evaluation games’. To do this I did a quick and dirty ranking of the teams each season. Then I isolated a set of 150 teams: tourney teams plus the next highest ranked teams until we reach 150. Evaluation games are games that happened in the tournament or the 4 weeks prior and were contested between two of the 150 teams. Additionally, I include the tourney games twice in the set of evaluation games to give them a bit more weight. This ends up amounting to around 500 games per season instead of the 63 if I was only using tournament games. Training of models can use games that are not evaluation games, but only these evaluation games are used for tuning hyper parameters, etc.

Many of the evaluation games are well before the tournament which poses a problem: we want to both respect the arrow of time, but also use all of the latest data when making predictions. To accomplish this, we train each model 5 times per year: each time using data including game data including dates up until N weeks before the tournaments--for N = 0, 1, 2, 3, 4. Then, when evaluating the evaluation games we use the most recent model that still respects the arrow of time. 

To fit various hyper parameters of the models I used a time-series cross validation strategy. For each year used in evaluation I would use the 8 prior years to train a model and then evaluate out of sample. Further, I used these out of sample models when fitting the ensemble coefficients.

**RESULTS**

Here’s how the various models would have fared individually. Subtract ~0.01 to simulate doing a 0/1 strategy on one of the first round games: 

Gaussian Latent Variable Model -- 0.4514
Gaussian Latent Variable Mode w/ Logistic at end -- 0.4050
OneShiningMGF-style -- 0.4162
Raddar-style -- 0.4380
Ensemble -- 0.4102
Ensemble with 0/1 ‘gambling’ -- 0.3998

So both the Gaussian Latent Variable Mode w/ Logistic at end and the OneShiningMGF-style models would have been good enough for gold if submitted alone. The ensemble wasn’t quite as good at the best individual model, but it wasn’t far off either.
