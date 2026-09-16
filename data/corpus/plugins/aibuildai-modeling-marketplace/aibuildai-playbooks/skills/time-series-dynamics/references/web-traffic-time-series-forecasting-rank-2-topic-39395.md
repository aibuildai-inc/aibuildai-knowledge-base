# My bet (2nd place)

Competition: web-traffic-time-series-forecasting
Rank: #2
Source: https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39395

I have absolutely no clue about how my approach will fare on stage 2, hence I'm sharing some of it before it shows embarrassingly bad on LB ;)  It gave good results on stage 1 with a public LB score of 41.4.  

It is based on 5 ideas.  

The first idea is to use the yearly seasonality of the data. There is a weekly seasonality, quite strong for some projects, e.g. es.wikipedia.com, and almost non existent with some others.   But, more importantly, there is a huge yearly seasonality.  

I therefore decided to train my models using cv on data from 12 months ago.  For stage 1, I trained with Jan 1, 2016 to Mar 1, 2016 as test set, building features out of July-Dec 2015 views.  Then I predicted values based on the same features made out of July-Dec 2016.  For stage 2 I trained using 62 days from Sept 13 2016, and same features built out of the data up to Sept 10 2016.  Then predicted using same features built out of data up to Sept 10 2017.   In order to remove the yearly trend and keep seasonality I substracted the median of the last 20 weeks of train data from the train and test data.  Once medians are substracted, value ranges are comparable year to year.  This deals with the fact that XGBoost cannot really extrapolate (see https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/38352).  

I used 5 fold CV where data was split by pages: all data for a given page must be in same fold otherwise the model overfits.  I took the median of the models trained on each fold for the submitted prediction.  Predictions were rounded to nearest integer with negative values set to 0.

This first idea is really a bet, and I have absolutely no clue about how effective it will be on stage 2.

The second idea is to not use RMSE and to approximate SMAPE with log1p transformed data.  SMAPE is about ignoring outliers.  Using a log1p transform of all views data is a first step towards ignoring outliers.  A second step is to use the right objective function for each of the three algorithms I used:

* Generalized linear model.  I used sklearn Huber regressor.  It is a linear regression where rmse (as in OLS) is replaced by Huber loss.  This is way less sensitive to outliers than OLS.  This yield a score of 43.9 in stage 1 public LB using 5 fold CV.

* XGBoost.  I defined a custom objective function for MAE rightaway (this was later suggested by Arthur Suilin).  Basically, gradient is positive if the prediction is above the true value, negative if prediction is below the true value, and 0 if they are equal.  I used a constant hessian with a high enough value (true hessian is null everywhere except when prediction equals true value).  Code is straightforward (there may be typos as I'm retyping from memory)

        from numba import jit
    
        @jit
        def grad(preds, dtrain):
            labels = dtrain.get_labels()
            n = preds.shape[0]
            grad = np.empty(n)
            hess = 500 * np.ones(n)
            for i in range(n):
                diff = preds[i] - labels[i]
                if diff &gt; 0:
                    grad[i] = 200
                elif diff &lt; 0:
                    grad[i] = -200
                else:
                    grad[i] = 0
        return grad, hess

    I later optimized it by pre allocating the arrays.  The constant 200 was selected by cv with a learning rate of 0.1.  Lower values were leading to very slow convergence.  Higher values were leading to worse CV score.  XGBoost settings were conservative, with min_child_weights set to 400.  400 was selected by CV.

    I got a stage 1 public LB of 42.6 with it,  using 5 fold CV.  Main issue with this model was its training time of 12 hours, because of the custom objective function (and the custom evaluation function).  Training with a built in objective is 10 times faster, but leads to way worse results.  I later found that I could get about the same accuracy by training it on 1/10 of the data.  This was key to meet stage 2 deadline!  

 * Neural nets.  I used a clipped version of MAE defined this way in Keras:
 
        import keras.backend as K
        
        def smape_error(y_true, y_pred):
            return K.mean( K.cplip( K.abs(y_pred - y_true), 0.0, 1.0), axis=-1)

 Clipping threshold was selected via CV.  I got a stage 1 public LB of 41.8 with it,  using 5 fold CV.  The NN is described in more details below.

The third idea is to get rid of outliers.  The best way I found was to use my best individual model submission and compute its smape row by row, and discard rows where smape is higher than 1.25.  The 1.25 threshold was selected by CV.  This moved my XGBoost model public LB score from 42.6 to 42.2.

Fourth idea is to ensemble everything in xgboost by training it on the residuals of the Keras predictions and the same features as my XGBoost model plus out of fold predictions from Huber regressor and Keras model.  I used a slightly modified version of the custom objective, where the gradient is inversely proportional to the square of the error.  This is the actual gradient of SMAPE.  This moved the score down to 41.4 from 41.8 with Keras only.

Fifth idea is to use medians as features instead of raw values, inspired by the very effective public kernels.  Features are mostly weekly medians for the past 8 to 12 weeks, depending on model, plus medians over larger periods.  One week lag of median, diff with median, max of week, were added later.

For Huber regressor and for XGBoost, rest of features are: day of week (Mon=0 to Sun=6), to capture weekly seasonality, week rank in test set, to capture weekly evolutions, and project (top url).  Using access and agent  did not help for XGBoost, but it did help a bit for NN. There are few additional features but their effect was quite marginal. I removed most of them in the last days of the competition anyway.  

For NN, I didn't explicitly used day of week and week rank.  I rather used them to define output indices: the output for the third Tuesday is at the same position, whatever the year I am in.  This was key for good predictions as it enables the prediction of weekly patterns.  Project and access/agent are one hot encoded, medians are the log1p transformed data.  I did not normalize data as mean is close to 0 already, and variance close to 1.  

The NN is a feedforward network with 200, 200, 100, 200 cells in each layer.  Input is concatenated again with the output of the first layer.  I don't know why but this boosted accuracy.  Activation is relu except for last one which is linear.   I used dropout of 0.5 for almost all layers (0.5 was selected by CV).  I also used a batch normalization for the middle layer. Model is compiled with adam optimizer and the loss function defined above. 

I used 5 fold Cv for NN  Batch size was huge: 4096.  I found that increasing batch size was both decreasing training time, and improving accuracy.   Training went for 200 epochs, computing actual training SMAPE every 10 epochs across all folds, and using median of all predictions of best epoch for submission.  For each fold I trained 1 model, and took the median of their predictions.  10 epochs take few seconds per model on my GTX 1080 Ti, thanks to the large batch size.

I tried CNNs and RNNs (LSTM, and Seq2Seq) but did not get results as good as the simple FF network.  No surprise as this is the first time I'm using deep learning.  I am eager to learn from those who successfully used CNN or RNN.

Another thing I tried was to ensemble XGBoost models.  It yield a 0.3 improvement but I could not run it for stage 2 submission because of the lack of time between data availability and deadline.

Hope you find this useful sharing.  I think (but hope not) that simpler approaches based on RNN will prove more effective in stage 2.  Let 's see.

Edited: My code is available on [github][1].


  [1]: https://github.com/jfpuget/Kaggle/tree/master/WebTrafficPrediction
