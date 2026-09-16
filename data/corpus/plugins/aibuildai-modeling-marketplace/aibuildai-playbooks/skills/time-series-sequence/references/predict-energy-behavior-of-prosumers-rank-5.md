# 5th place solution

Competition: predict-energy-behavior-of-prosumers
Rank: #5
Source: https://www.kaggle.com/c/predict-energy-behavior-of-prosumers/discussion/499938

### A few words about myself

I completed my PhD in AI in 2011, however from there, I opted for travel and adventure, rather than a career in R&D. Only in recent years I got back to ML, and despite all the buzz around AI, I'm still having a hard time finding a job. Perhaps living in Thailand as an expat complicates things a bit, so I resort to freelancing and that's also why I had plenty of time to dedicate to this competition. I enjoyed it a lot, first, because it was a great lesson, and also, because it was easy on hardware and thus allowed for plenty of experimentation. 


### Solution overview

After trying out some regression models from `scikit-learn`, out of curiosity, I turned to XGBoost and finally to LightGBM which gave the best baseline just out-of-the-box. I didn't have prior pleasure with Gradient Boosting methods in a real-world problem setting, so it was even more interesting. I came to conclusion that learning from notebooks shared by more experienced players is quite effective, when we have examples of how things can be done elegantly, such as [Enefit│XGBoost│starter](https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter) and [Enefit Generic Notebook](https://www.kaggle.com/code/greysky/enefit-generic-notebook/notebook) (I hope and believe these are the origin of the code which turns up modified-or-not in many other notebooks later on.) It was also the first time I saw `polars` in action. Thank you gentlemen for showing the way!

As everybody knows, the problem was largely about feature selection and engineering. So it took lots of experimentation, in fact the whole process was one big loop akin to trial & error: idea & evaluation. The `MLflow` package was particularly helpful in that. Obviously I don't even remember all the ideas that I have tried out, so I will only describe what survived. More details on features in the next section.   

When it comes to the model, I opted for LightGBM and from there not much can be done but fine-tuning the parameters and eventually configuring an ensemble. Fine-tuning was done using a simple custom evolutionary search and a 2-stage cross-validation scheme. The 1st stage was time series cross-validation with 3 folds and each fold holding 2 months of testing data, so:

- training data until 2022-09-01, testing until 2022-11-01
- training data until 2022-11-01, testing until 2023-01-01
- training data until 2023-01-01, testing until 2023-03-01
###  
This split was used primarily for intense algorithmic tuning and feature selection. The 2nd stage, i.e. a simple split with the remaining 3 months of data (until 2023-05-26) I used for less intense manual tuning and selection. The final model selection was naturally based on the public test data. Interestingly and admittedly, I **was** trying to beat the score on the public LB, but somehow I wasn't able to, ending up at 22nd place. Perhaps my local CV restricted me a bit in this pursuit, but anyway, I consider my entry to be very lucky.

By looking at some other solutions, I think my proposal is more on the *simple* side. Crucially for the performance, "production" and "consumption" units are tackled separately. The production part is fed with only 75 features, while the consumption with 85. A scikit-learn's `VotingRegressor` ensemble of 3 LightGBM regression estimators is trained for each part, with slightly different parameters. Specifically:

```
                                 Production          Consumption

        boosting_type:                        gbdt
        num_leaves:                            358
        max_depth:                              10
        learning_rate:                0.02                0.025
        min_split_gain:                0.0                  0.1
        min_sum_hessian_in_leaf:       1.0                  0.1
        min_data_in_leaf:              120                  360
        bagging_fraction:                      1.0
        bagging_freq:                            0
        feature_fraction:                      0.8
        feature_fraction_bynode:               0.9
        lambda_l1:                            0.01
        lambda_l2:                             1.0
        max_bin:                               255
        n_estimators:                         2000
```

As you might have guessed, I cannot really justify those values -- they are just a result of fine-tuning. All training data is used at the submission evaluation with no validation set for early stopping. 2000 estimators didn't seem to give much advantage over let's say 1200, but on the other hand I didn't observe any heavy tendency for over-fitting at cross-validation, and also there was more data coming, so 2000 looked fine.

Finally, important part of the solution is online re-training, which is done every 9 days of scoring data, so about 10 trainings along evaluation in total. As training of both ensembles is somewhat costly -- it takes about 20 minutes with GPU -- it was important to train only for scoring periods. Incoming data had to be collected for each day regardless. It was critical to understand how the submission evaluation works at all stages exactly and be careful to avoid unforeseen errors. I don't know if anyone's solution fell out at this turn, but it felt a bit slippery, for instance when switching between pandas and polars. I even encountered errors when using just different versions of these great libraries. Yet, maybe hosts took proper care to secure an easy pass for any solution passing the public data evaluation. I'm sure it could be very nasty if there were glitches in testing data.


### Feature selection and engineering

Regarding features, I often wondered which approach is better: reduction or redundancy. I tended to reduce the number of features and simplify stuff wherever possible, but the dilemma comes back regardless. I mean, when a feature or a set of features seem to have little effect at cross-validation, do you keep it or remove it? The top solution by hyd apparently takes a very generous approach here, with lots of "lagging" features and reportedly not much care of what goes in, but was it actually for the better? LightGBM models can do their own feature selection and can handle random and useless features easily, however I imagine, in a multitude of features there is also risk of picking up "false patterns". Ultimately, it probably just depends on the problem and the features.

For a while, I was using `Optuna` for feature selection, however later on, I was more comfortable with a simple custom evolutionary search, which worked for the model fine-tuning as well.

Because there aren't too many of them, I can list all the selected features, for production:
```
county, is_business, product_type, installed_capacity, euros_per_mwh,

hours_ahead_fd, temperature_fd, dewpoint_fd, cloudcover_high_fd, cloudcover_low_fd, cloudcover_mid_fd, cloudcover_total_fd, 10_metre_u_wind_component_fd, 10_metre_v_wind_component_fd, direct_solar_radiation_fd, surface_solar_radiation_downwards_fd, total_precipitation_fd,

temperature_fl, dewpoint_fl, cloudcover_high_fl, cloudcover_low_fl, cloudcover_mid_fl, 10_metre_u_wind_component_fl, 10_metre_v_wind_component_fl, direct_solar_radiation_fl, snowfall_fl, total_precipitation_fl,

temperature_hl, dewpoint_hl, rain_hl, snowfall_hl, surface_pressure_hl, cloudcover_total_hl, cloudcover_low_hl, cloudcover_mid_hl, cloudcover_high_hl, windspeed_10m_hl, winddirection_10m_hl, direct_solar_radiation_hl, diffuse_radiation_hl,

dewpoint_fl_dmean, cloudcover_high_fl_dmean, cloudcover_low_fl_dmean, cloudcover_mid_fl_dmean, cloudcover_total_fl_dmean, 10_metre_u_wind_component_fl_dmean, 10_metre_v_wind_component_fl_dmean, direct_solar_radiation_fl_dmean, surface_solar_radiation_downwards_fl_dmean, snowfall_fl_dmean, total_precipitation_fl_dmean, hours_ahead_fl_dmean, temperature_fl_dmean,

target_6r_2, target_6r_3, target_6r_4, target_6r_5, target_6r_14, day, weekday, sin_dayofyear, cos_dayofyear, cos_hour, sin_hour, target_mean_ic_ratio_2, target_std_6r, target_delta_68_6r, target_7r_2, is_holiday, is_holiday_2, n_holidays_past_week, ssrd_t, ssrd_t_mean
``` 

and consumption (diff):

```
+ target_2, cloudcover_total_fl, month, surface_solar_radiation_downwards_fl, target_4r_14, snowfall_fd, eic_count, target_delta_68_7r, target_4r_7, target_ratio_18_7r, installed_capacity_ratio, target_7r_7, target_std_7r, hours_ahead_fl, target_mean_2, target_4r_2, shortwave_radiation_hl, target_ratio_68_7r, year

- target_6r_2, target_6r_3, target_6r_4, target_6r_5, target_6r_14, day, cos_dayofyear, ssrd_t, ssrd_t_mean
```

Notably, the gas price has been dropped from both sets, as well as the country-wide historical weather. But instead, we have a bunch of `*_fl_dmean` features which are differences between "forecast local" weather and their average over the past week. There's also a couple of `target_*` features which are some transforms of the revealed targets (lags, ratios, averages, differences) and which survived the selection. The `ssrd_*` are based on the transform suggested [here](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/discussion/468654). `is_holiday_2` is true if 2 days before was a holiday (maybe useful in conjunction with those past targets transforms that involve the most recent revealed target value, i.e. from 2 days before.) 

Another always-trailing dilemma were date-related features, like `day`, `month` or `year`, often on the border of significance at cross-validation. Having the trigonometric transform of the day of year should make `day` and `month` redundant, but of course it doesn't work like that. The biggest mystery though was the `year`, which in our case had very limited span. Sometimes you see general advice to keep such a feature, but honestly I can't figure out how a model can make use of a feature that is going to have a new, previously unseen value at prediction. Maybe it's already helpful for the model to see different years during training? It would be nice to find the right explanation. 

A big part of the solution and closely related to feature engineering was "target transformation". I did plenty of trials here, but ultimately I went with two simple transforms, for the production units:
```
    z = target / (1 + installed_capacity),
```
where `installed_capacity` is obviously the most recent value available at prediction time, so lagging 2 days. And for the consumption units:
```
    z = target / (1 + target_avg),
```
where `target_avg` is the average target from the most recent available 7 days (so actually day `d-2` up to `d-9`), sampled at the same time as `target`, which is also from day `d-2`. Interestingly, I tried to train the models assuming "perfect information", i.e. without the lags, as if all the information about the current datetime was available, but curiously it didn't make much difference to prediction. Anyway, there was additional processing and NaN filling to avoid any errors. The extra `1` in denominators is to prevent excess values in those hypothetical glitchy cases when the other term would be close to zero (it could be just clipping, too, though.) One more treatment which I will never know was ultimately helpful or not, was filling any missing values of the transformed target. The filling values were taken as median of similar units across the country. I guess it could be helpful during prediction for any unforeseen or interrupted prediction units, although according to cross-validation, filling up with just moderate constants was equally good. 


^^
That's it, big thanks to the hosts for this great competition, congrats to the winners, cheers to all players, comments welcome and till next time!
