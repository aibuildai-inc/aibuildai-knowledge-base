# 3rd Place - Predict Multipliers with GRU

Competition: godaddy-microbusiness-density-forecasting
Rank: #3
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/418287

Thank you Kaggle and GoDaddy for an exciting forecasting competition!

# Predict Multipliers with TensorFlow GRU
Our forecasting models need to predict 3135 county microbusiness density predictions per future month. The public LB is one future month and the private LB is three future months (with 2 month gap between train and test).

From Kaggle's M5 competition [here][1] we learned that predicting multipliers (between the test time period and train time period) was most important. For example in M5, we could take an average public notebook and multiply all predictions by `0.95` and win Gold medal !! Similarly in Kaggle's GoDaddy competition, a simple last value baseline multiplied by the correct January/December multiplier of `1.0045` as shown in Vitaly's notebook [here][2] achieved Gold medal public LB for many months!

To predict multipliers, we convert train data into multipliers and then train a TensorFlow GRU. This simple model achieves 15th place Gold. Then if we post process the multipliers based on ratios learned from LB probing, we boost our solution to 3rd place Gold !!

# Original Train Data
Kaggle gave us 41 historical months of data for each of 3135 USA counties. 



# Step 1 - Adjust Train Data

The definition of microbusiness density is `micro businesses per 100 people over age 18`. The value of microbusiness density changes when county population changes. So the first step is to convert all microbusiness density to use the same 2021 census. We use the formula

    adjusted_microbusiness_density = microbusiness_density * (population / population_year_2021)

# Step 2 - Create 56,000 Time Series!
For each county, we create 18 time series. We use 13 consecutive months to train our GRU and predict the next 5 months. (Pictured below is only 3 time series per county for illustration purposes). Afterward, we have 56,000 time series to train with!



# Step 3 - Use Only Top 90% Largest Counties
We notice that the smallest 10% counties have nearly the same microbusiness density from month to month, so we only train our GRU using the largest 90% counties. During inference, we will use the last known value as prediction for small counties.



# Step 4 - Convert to Multipliers
Each county has 13 months of train and 5 months of valid. We will convert the raw micro business data into ratios by replacing each value with the ratio of current month divided by previous month. Afterward we will have 12 months of train ratios and 5 months of valid ratios

    # CONVERT TO RATIOS
    for k in range(17):
        new_data[:,k+1] = old_data[:,k+1] / old_data[:,k]

# Step 5 - Train GRU with GroupKFold
When creating KFold we need to use GroupKFold and keep all time series for each county (from the possible 18 time series per each county) within the same fold. Otherwise CV score will be inaccurate and we cannot optimize our models hyperparameters correctly. Below is our TensorFlow GRU model which takes an input of 12 ratios and predicts 5 ratios:

    def build_model():
    
        inp = tf.keras.Input(shape=(12,1)) # INPUT SHAPE IS 12
        x = tf.keras.layers.GRU(units=8, return_sequences=True)(inp)
        x = tf.keras.layers.GRU(units=8, return_sequences=True)(x)
        x = tf.keras.layers.GRU(units=8, return_sequences=False)(x)
        x = tf.keras.layers.Dense(5,activation='linear')(x) # OUTPUT SHAPE IS 5
        model = tf.keras.Model(inputs=inp, outputs=x)
    
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss, optimizer = opt)
    
        return model

# Step 6 - Infer and Post Process
We predict each county individually. To make predictions for January 2023. We begin with the last known value (of the county we are predicting) from December 2022 and multiply by the first ratio predicted by our model. To predict Feb 2023, we take our Jan 2023 prediction and multiply by the second ratio predicted by our model. To predict Mar 2023, we multiply Feb 2023 by third ratio. To predict Apr 2023, we multiply by fourth ratio. And finally to predict May 2023, we multiply by fifth ratio.

We can improve our predictions by modifying the first ratio (which is the ratio of January 2023 divided by December 2022) by information from probing the public LB.  Probing informs us that the best January/December average ratio for the largest 90% counties is `1.0045`. Similarly, we can improve our predictions for small counties by probing the public LB to find ratios for different sized small counties.

Post process boosts our GRU solution from 12th place Gold to 3rd place Gold !!

# GRU Solution Code Published
Full solution code with preprocess, train, and infer is published [here][3]. 

# More Solution Models
My final two submissions were
* GRU with PP - 3rd place
* Linear Model with PP - 10th place

More information about my linear model is [here][4]. Enjoy!

[1]: https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163621
[2]: https://www.kaggle.com/code/vitalykudelya/21-lines-of-code
[3]: https://www.kaggle.com/code/cdeotte/gru-model-3rd-place-gold
[4]: https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/discussion/395098
