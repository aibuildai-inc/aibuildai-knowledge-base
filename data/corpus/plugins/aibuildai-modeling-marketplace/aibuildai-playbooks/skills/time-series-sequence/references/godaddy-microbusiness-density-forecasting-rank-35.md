# 35th Place Solution - GRU+XGB

Competition: godaddy-microbusiness-density-forecasting
Rank: #35
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/418657

Goal of this competition was to predict monthly ~~microbusines density~~ [domain registrations of small businesses](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/374293) divided by the active population for all counties in the US. The biggest difference to other timeseries competitions is in my opinion that the public leaderboard tracked the score of the one month forecast (January) while our final score would be determined on 3-5 months in the future (March, April, May). My expectation for this competition was to deal with various macroeconomic explanatory variables and maybe even include some capital markets data. As it turned out this was not the case😄.

### Things that did not work
- I spent more time than I'm willing to admit on scrolling through files from statsamerica (www.statsamerica.org/downloads) trying to find anything helpful to enhance the data we had at hand. Unfortunately I can't present any evidence for it's usefulness in this context. Anyway they have cool stuff (https://www.statsamerica.org/innovation) over there.
- Capital markets indicators also had little effect. The decision to register a domain is surprisingly not driven by the VIX or FOMC dots (or more precise it does not help on county level forecasting).

### Data Quality and Preprocessing
Instead, outlier detection was the name of the game. I didn't probe the leaderboard but nevertheless spent much time comparing small counties and their erratic timeseries. Additionally, we had to deal with a [structural change](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/375715) in the data in January 2021 and the [underlying population numbers](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/389215) were also shifting. 

Therefore the target variable was adjusted for the latest census data and I removed suspect jumps for all counties in this month if they exceeded a threshold of +-7%:
```python
for o in tqdm(raw.cfips.unique()):
        indices = (raw['cfips']==o)
        tmp = raw.loc[indices].copy().reset_index(drop=True)
        var = tmp.microbusiness_density.values.copy()
        var_pct = tmp.microbusiness_density.pct_change().clip(-0.07,0.07)
        for j in range(40, 0, -1):
            if j==18 and (var_pct[j]== -0.07 or var_pct[j]== 0.07):
                var[j-1] = var[j]
            else:
                var[j-1] = var[j]/(1+var_pct[j])
        raw.loc[indices, 'microbusiness_density'] = var
```
Afterwards the timeseries was further denoised with an autoencoder:
```python
def create_autoencoder(noise=0.05):
    i = tf.keras.Input(shape=(24,))
    encoded = tf.keras.layers.BatchNormalization()(i)
    encoded = tf.keras.layers.GaussianNoise(noise)(encoded)
    encoded = tf.keras.layers.Dense(64,activation='relu')(encoded)
    decoded = tf.keras.layers.Dropout(0.2)(encoded)
    decoded = tf.keras.layers.Dense(24)(decoded)
    x = tf.keras.layers.Dense(32,activation='relu')(decoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(24,activation='linear')(x)
    
    encoder = tf.keras.Model(inputs=i,outputs=x)
    loss = tf.keras.losses.MeanSquaredError()
    encoder.compile(optimizer=Adam(0.001),loss=loss)
    return encoder
```
I'm unsure if this could add any kind of leakage as it was applied before/outside of the CV fold structure. But it stabilized my CVs and had no negative impact on the public leaderboard score. Anyway I kept using it only for the GRU model and trained XGB on the 'raw' data.

### Models and Target
Only counties with more than 150 microbusinesses were considered as input for the model. For counties below this threshold the last value was used.
Building on top [Chris proposal](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/381038#2114357) I used GRU with 24 months sequence of np.log1p(microbusiness density) input combining 3 folds (GroupKFold grouped by date) and forecasting 1, 3, 4, 5 months into the future.
As a second model I used XGB derived from what [GIBA shared](https://www.kaggle.com/code/titericz/better-xgb-baseline), defining the 1, 3, 4, 5 months growth difference as a target. Most of the county level features were included here but I think they added little explanatory value. For postprocessing I clipped the XGB predictions of the percentage changes with the corresponding 10% and 90% quantile values observed after preprocessing. Both models are weighted with 50% in the final submission.

### Second Submission
This was a simple linear model inspired by the success of [21 lines of code](https://www.kaggle.com/code/vitalykudelya/21-lines-of-code). It calculated the average monthly growth per county on the 21 months after the structural break in January 2021. The mean was clipped at -0.5% and +1% and for counties with less then 150 active domains this value was scaled by 0.3 (all of this was expert judgement based on very crude statistics, average growth, 5%,25%,75%,95% quantiles of growth). For counties with stale growth in November and December the last value was forecasted perpetually. This scored 4.0731 which would have been place 175 (still a silver medal with 10% of the effort). 🙀
