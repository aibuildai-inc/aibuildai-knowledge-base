# 1st place - I think it was genetic programming

Competition: playground-series-s5e10
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming

Congratulation to all the winners and many thanks to those contributing quality discussions and code: @cdeotte, @siukeitin, @optimistix, @mikhailnaumov, @mahoganybuttstrings You know who you are even if I forgot to mention you here.

I want to credit in particular @masayakawamata and his [**excellent TabM notebook**](https://www.kaggle.com/code/masayakawamata/s5e10-single-tabm-tuned). It helped my ensemble, but mostly because I learned to use a new group of neural networks that I suspect will be useful in the future.

It was a close competition on public LB, and remained so on private LB. Other than the top 4 spots "tied" at `0.05563`, the next ~200 competitors were all at `0.05564`. We've had it close before, but this seems extreme even by Kaggle standards.

In my academic calendar I have a somewhat open three-week stretch in October, so this is my only chance to compete meaningfully outside of summer and winter breaks. So I did, and my computers could attest to that. It seemed to me from the beginning that not much progress would be made in this competition after the first week or so. Boy, did I underestimate it! Forget about "not much progress", how about no progress at all? I have had the same LB score for the past 3 weeks, even though it was internally improving by miniscule amounts when I sorted about a dozen submissions that scored `0.05537`. As scores from other Kagglers remained the same for the most part, it was clear that this stagnation was not happening only to me.

I tried many things, and it would take me a long time to summarize. Instead, I will mention what I think made the most difference, even though I am not sure there was a difference given that 3 Kagglers are tied with me, and another 200 are behind by only 0.00001.

[Flowchart]

I meant to make a helpful chart but the result above seems overly complex. Hopefully it will do the trick. The first row is various data types that were used. `Mix` refers to original data that had both categorical and numerical features. Sometimes they were all converted to `Categorical` or to `Numerical`. After about 10 days I added latent space representations by `Autoencoder` and some features derived by `Genetic programming`.

Purely numerical representation didn’t work, or I should say it didn't work by Kaggle standards. For example, I got a 0.058 CV score and a 0.05802 private score from Lasso, which is objectively excellent for a method that takes about 2 minutes for a 10x10 repeated K-fold run. Not only that, but Lasso feature coefficients were extremely informative in this competition.

[Lasso works]

When you translate what’s in that image into numbers (while ignoring coefficients with absolute values < 0.05), you get this:

```
accident_risk = 
0.3046 * curvature +
0.0706 * lighting_2 +
-0.1203 * lighting_0 +
-0.1181 * lighting_1 +
-0.0928 * weather_0 +
-0.1232 * speed_limit_0 +
-0.1232 * speed_limit_2 +
-0.1218 * speed_limit_1 +
0.0646 * speed_limit_4 +
0.0629 * speed_limit_3 +
0.0789 * num_reported_accidents_4 +
0.0704 * num_reported_accidents_5
```

Compare that to the original equation used to create the target, and we have a strong Lasso model that clearly explains feature contributions.

```
base_risk = (
    0.3 * data["curvature"] + 
    0.2 * (data["lighting"] == "night").astype(int) + 
    0.1 * (data["weather"] != "clear").astype(int) + 
    0.2 * (data["speed_limit"] >= 60).astype(int) + 
    0.1 * (np.array(data["num_reported_accidents"]) > 2).astype(int)
)
```

But enough of that. As you can see from the flowchart above, I tried many different neural networks. Got introduced to PyTorch Tabular and even learned rudimentary PyTorch programming. None of those "worked" by Kaggle standards. That’s where the TabM notebook was so valuable, because it gave another model that could work with my Keras FM and TF embedding networks. I added XGB residual boosting by @cdeotte too late to my ensemble, and by that time neither my CV nor LB scores were budging regardless of promising individual models. In the end I fine-tuned TabM parameters and got three models that had better scores than the original, and those parameters can be found in [**this notebook**]( https://www.kaggle.com/code/tilii7/s5e10-tabm-tuned-further). One of the three parameter sets is used, and the other two are commented out. You may want to try all three and see if that helps your ensembles. XGBoost and CatBoost were fairly standard for me in this competition, didn’t use LightGBM except for a couple of runs at the start.

I do ensembling with multiple methods because it is generally easy, and one never knows what will work. That said, my preference is with NNs and Lasso for regression, followed by hill climbing for getting a quick result. That’s because all these methods generally require no tunning. You will see from the flowchart that I also used XGB and CatB for ensembling, but those are long Optuna runs where finding the optimal parameters takes 4-10 hours. After about 2 weeks I was ensembling exclusively with NNs as it worked better than anything else and was reasonably fast (1-2 hours for 40-50 models).

I did several AugoGluon runs that produced many promising models, but none of them were helping the ensemble.

Since I didn’t talk about it up in the data section, I should say here that neither autoencoders (AE) not genetic programming (GP) produced features that worked well on their own. Again, take this as a relative assessment, because they worked better than Lasso, but were generally not competitive. Adding them to individual models didn’t help much either.

Here is what I think made a little difference: adding 11 GP-derived features at the ensembling stage. I have a notebook [**here**]( https://www.kaggle.com/code/tilii7/s5e10-genetic-programming-features) showing how the GP features were derived. As they are added to the original data, you can simply copy those files and try them in your pipelines. These features are difficult to rationalize by looking at the formula:

```
train['GP_11'] = np.sqrt(np.abs(np.sin(np.sin(np.sin(np.sin(np.sin(np.sin(0.254397*train['speed']*np.sqrt(np.abs(train['lighting']*train['speed']*np.exp(train['curvature'])*np.sqrt(np.abs(train['lighting']**3*np.sin(train['speed']**2*train['accidents']*np.cos(train['weather']))/train['weather']))))))))))))
```

Notice, however, that even here only the relevant features were selected for calculation. Just for fun, I did a couple of GP runs that disallowed all the fancy algebra and trigonometric manipulations, and only [+, -, *, /] operations were allowed. Check out what two of those runs produced independently.

```
0.308*[curvature] + 0.180*[speed_limit_3] + 0.180*[speed_limit_4] + 0.128*[num_reported_accidents_4] + 0.180*[lighting_2] + 0.107*[weather_1] + 0.118*[weather_2] - 0.0004
```
```
0.309*[curvature] + 0.186*[speed_limit_3] + 0.186*[speed_limit_4] + 0.0929*[num_reported_accidents_4] + 0.186*[lighting_2] + 0.099*[weather_1] + 0.0929*[weather_2] + 0.011
```

One would think that these would be great extra features because they recapitulate the original target equation almost perfectly, but that’s not the case. As they are so similar to good models and individually have ~0.059 RMSE to the target, it seems that they are not really needed at the ensembling stage. Basically, they don’t bring diversity. Other GP features in that notebook were generally more useful in my ensembles. Specifically, adding 11 GP features on top of individual models resulted in a 0.00001 ensemble improvement.

The last improvement came from using CatBoost as a secondary ensembler. CatB has the ability to use a strong “baseline” model as a seed before ensembling, which is essentially the same as boosting residuals except that adding and subtracting from the target is done automatically for us. So I would first run a Keras ensemble without 11 GP features, then run CatB with the same group of models + 11 GP features and use the Keras ensemble model as a baseline. This improved both pubic and private LB scores by ~0.00002. That and GP features is what I suspect made the difference in the end.

The final step was taking several first-level ensemble models and making a second-level ensemble by hill climbing. Although, one could argue that they were already second level ensembles because I run them first through Keras and then CatB. That final hill climbing ensemble (using 4 previous ensemble models) finished with the best score and was my winning solution, even though it looks the same over 5 decimals as the best individual ensemble. In fact, all my top 12 solutions look the same over 5 decimals, so I am happy that among them I picked 2nd and 5th best solutions.
