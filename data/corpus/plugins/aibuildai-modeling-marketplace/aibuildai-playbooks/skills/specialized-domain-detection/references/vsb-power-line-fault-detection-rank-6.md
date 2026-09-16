# 6th Place Solution Overview

Competition: vsb-power-line-fault-detection
Rank: #6
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/85170#latest-500367

Thanks to VSB/Enet Centre and Kaggle for this great competition. I had a lot of fun with that in the last months!

I'm going to brief the main points I believe have helped in my final score. I'm not a Pro (yet!), so if you think there is something incorrect or that could be improved, please leave your comments!

**My final solution was a ensemble of three main branchs:**

1. Gradient Boosting Trees (Lightgbm)
2. Recurrent Neural Network (GRU and LSTM)
3. Convolutional Neural Network (Custom CNN, Resnet50, DenseNet101)

**Preprocessing:**
The 3 signal phases were used as just one sample. Each signal was aligned (using the 50Hz phase of the fourrier transform) to start where the signal crosses the axis from the negative to positive (𝜋/2) and the last quarter of the signal was removed.
In each model, I used a few denoised versions (varying thresholds) of this aligned/cropped signal:
1. Wavelet
	- Following the MaxHalford's repository: [extract\_solo\_features.py](https://github.com/MaxHalford/kaggle-vsb-power/blob/master/scripts/extract_solo_features.py)
2. Fourrier / IQR
	- Eliminate low frequencies with Fourrier transform and points with interquartile range

I ended up with 4 signal version (Wavelet/Fourrier with 2 threshold levels)

To RNN and CNN, I undersampled the 600000 size signal to 300000 taking the position with maximum absolute value at each pair of points:
```und_signal = np.where(np.abs(signal[::2])&gt;np.abs(signal[1::2]), signal[::2], signal[1::2])```


**Inputs:**
1. GBT:
	- The features were min, max, mean, std, skew and kurtosis of:
		- Peaks count, height, width, prominences: [extract\_solo\_features.py](https://github.com/MaxHalford/kaggle-vsb-power/blob/master/scripts/extract_solo_features.py)
		- Entropy &amp; Fractal: [vsb-competition-attention-bilstm-with-features](https://www.kaggle.com/tarunpaparaju/vsb-competition-attention-bilstm-with-features)
		- Slope of lines connecting positive/negative peaks to the next negative/positive peaks
		- Ratio of positive peaks to the minimum of the next maxDistance (defined in Vantuch's Thesis) points.
		- Ratio of negative peaks to the maximum of the next maxDistance points.
	- R² and weights of some polynomial regressions fit in positive/negative peaks

2. RNN:
	- Preprocessed signal reshaped in 40 columns

3. CNN:
	- Preprocessed signal reshaped in 200 columns

**LB Prediction**
At some point in the competition, I built a Lasso to predict the LB score of a submission. I knew that it would create problems with overfitting so I tried to avoid it with:
- Threshold selection: I fixed all thresholds at the value 0.5
- Pseudo-Labelling: I used it in all models and I reduced the sample weights of test examples witch had low weight in the trained Lasso
- Ensemble: I should have ended up with 24 models (6 models x 4 preprocessed signals) but I got just 16 (lack of time...!)

I was not able to create a stable stacking so i used a very elaborate strategy to ensemble the models: Take the arithmetic mean of the predictions...

The last but not the least, I wanted to thank you everybody for the great kernels and discussions. I learned a lot and got my first gold medal!

That's all folks!
