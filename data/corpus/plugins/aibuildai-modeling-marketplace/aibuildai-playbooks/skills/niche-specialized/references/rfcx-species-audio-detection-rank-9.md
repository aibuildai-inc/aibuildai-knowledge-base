# 9th Place - Post Process LB 0.926 to LB 0.963!

Competition: rfcx-species-audio-detection
Rank: #9
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220389

Thanks Kaggle and RFCx for a fun competition. My final submission without post process achieves private LB 0.926 and with post process achieves **private LB 0.963**! That's +0.037 with post process!

# How To Score LB 0.950+
The metric in this competition is different than other competitions. We are asked to provide `submission.csv` where each row is a test sample, and each column has a species prediction.

In other competitions, the metric computes **column wise AUC**. In this competition, the metric is essentially **row wise AUC**. Therefore we need each column to represent probabilities. The columns of common species need to be large values and the columns of rare species need to be small values.

# Train Distribution
The distribution of the train data has roughly 6x the number of false positives versus true positives for each species. If you train your model with that data, then when your model is unsure about a prediction, it will predict the mean that it observed in the train data which is 1/7. Therefore if it is unsure about species 3, it will predict 1/7 and if it is unsure about species 19, it will predict 1/7.

This is a problem because species 3 appears in roughly 90% of test samples whereas species 19 appears in roughly 1% of test samples. Therefore when your model is unsure , it should predict 90% for species 3 and 1% for species 19.

# Test Distribution - Post Process
In order to correct our model's predictions we scale the odds. Note that scaling odds doesn't affect predictions of 0 and 1. It only affects the unsure middle predictions. 

First we convert the `submission.csv` column of probabilities into odds with the formula
$$ \text{odds} = \frac{p}{1-p}$$
Then we scale the odds with
$$ \text{new odds} = \text{factor} * \text{old odds}$$
And lastly we convert back to probabilities
$$ \text{prob} = \frac{\text{new odds}}{1 + \text{new odds}}$$

# Sample Code
    # CONVERT PROB TO ODDS, APPLY MULTIPLIER, CONVERT BACK TO PROB
    def scale(probs, factor):
        probs = probs.copy()
        idx = np.where(probs!=1)[0]
        odds = factor * probs[idx] / (1-probs[idx])
        probs[idx] =  odds/(1+odds)
        return probs

    for k in range(24):
        sub.iloc[:,1+k] = scale(sub.iloc[:,1+k].values, FACTORS[k])

# Increase LB by +0.040!
The only detail remaining is how to calculate the `FACTORS` above. There are at least 3 ways.
* Create a model with output layer sigmoid, not softmax. Train with BCE loss using both true and false positives. Predict the test data. Compute the mean of each column. Convert to odds and divide by the odds of training data.
* Probe the LB with `submission.csv` of all zeros and one column of ones. Then use math to compute `factor` for that species. UPDATE use random numbers less than 1 instead of 0s to avoid sorting uncertainty.
* Use the factors listed in RFCx's paper [here][1], Table 2 in Section 2.4

Personally, i used the first option listed above. I didn't have 5 days to probe the LB 24 times for the second option. And I didn't find the paper until the last day of the competition for the third option.

My single model scores private LB 0.921 without post process and scores private LB 0.958 with post process. Ensembling a variety of image sizes and backbones increased the LBs to 0.926 and LB 0.963 respectively.

# Model Details
I converted each audio file into Mel Spectrogram with Librosa `feature.melspectrogram` and `power_to_db` using sampling rate `32_000`, `n_mels = 384`, `n_fft=2048`, `hop_length=512`, `win_length=2048`. This produced NumPy arrays of size `(384, 3751)`. I later normalized them with `img = (img+87)/128` and  trained with random crops of `384x384` which had frequency range 20Hz to 16_000Hz and time range 6.14 seconds. Each crop contained at least 75% of a true positive or false positive.

I concatenated the true postive and false positive CSV files from Kaggle. My dataloader provided labels, masks, and one color images. The labels and masks were contained in the vector `y` which was 48 zeros where 2 were potentially altered. For species `k`, the `kth` element was 0 or 1 corresponding to false positive or true positive respectively. And the `k+24th` element was 1 indicating mask true to calculate loss for the `kth` species. I trained TF model with the following loss

     def masked_loss(y_true, y_pred):
    
         mask = y_true[:,24:]
         y_true = y_true[:,:24]
      
         y_pred = tf.convert_to_tensor(y_pred)
         y_true = tf.cast(y_true, y_pred.dtype)
         mask = tf.cast(mask, y_pred.dtype)
         y_pred = tf.math.multiply(mask, y_pred, name=None)

         return K.mean( K.binary_crossentropy(y_true, y_pred), axis=-1 )*24.0

I used EfficientNetB2 with `albu.CoarseDropout` and `albu.RandomBrightnessContrast`. The optimizer was Adam with learning rate `1e-3` and reduce on plateau `factor=0.3`, `patience=3`. The final layer of the model was `GlobalAveragePooling2D()` and `Dense(24, activation='sigmoid')`. I monitored `val_loss` to tune hyperameters.

# Try PP on Your Sub
If you want to try post process on your `submission.csv` file, I posted a Kaggle notebook [here][2]. It uses the 3rd method above for computing `FACTORS`. It also has 3 `MODE` you can try to account for different ways that you may have used to train your models.

[1]: https://www.sciencedirect.com/science/article/pii/S1574954120300637
[2]: https://www.kaggle.com/cdeotte/rainforest-post-process-lb-0-970
