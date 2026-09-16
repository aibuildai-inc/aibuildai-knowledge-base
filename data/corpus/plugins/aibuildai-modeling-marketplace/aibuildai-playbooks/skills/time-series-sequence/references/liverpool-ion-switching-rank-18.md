# 18th place solution

Competition: liverpool-ion-switching
Rank: #18
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153826

I'm a lucky person, and I won 18th place and a silver medal.
My solution is just a slight modification to the public wavenet notebook.

feature engineering
I used Data Without Drift from @cdeotte for the data.
42 features were used, including 3 lag features as well as @siavrez's wavenet notebook, signal**2 and signal**4 features based on @ragnar123's notebook, gradient, low-pass and high-pass filters based on @martxelo's notebook, and the addition of @sggpls' ION-SHIFTED-RFC-PROBA.

network architecture
The convolutional 16 base filters were changed to 24 (1.5x),And 3 LSTMs were added.

augmentation strategy and others
I normalized the features, added np.random.normal(loc=0, scale=0.01) to all the features, and flipped the data three times to make the training data 8x.
GROUP BATCH SIZE (signals per sample) was determined to be the best at 6250.

training
The training was conducted twice. Optimizing for losses as normal at first. For the second time, I changed the weight regularization of L2 to factor from 0 to 0.0001 and LR to 1e-5. And optimize accuracy.

prediction
Predictions were also augmented. I added np.random.normal(loc=0, scale=0.001) to all the features and flipped the data 4 times, i.e., to predict the test data 10 times.
In addition, I changed the GROUP BATCH SIZE from 6250 to 500000 13 times and averaged it and then used argmax.
The final oofCV was 0.942421.

Again, I'm just a lucky person. Excellent discussion and notebook sharing made me won a medal. Without these, my score would have been 0.
Thank you to those who shared insights.

Reference. They are my heroes.
datasets
ION-SHIFTED-RFC-PROBA(https://www.kaggle.com/sggpls/ion-shifted-rfc-proba)
Data Without Drift(https://www.kaggle.com/cdeotte/data-without-drift)
notebooks
WaveNet-Keras(https://www.kaggle.com/siavrez/wavenet-keras)
Wavenet with 1 more feature(https://www.kaggle.com/ragnar123/wavenet-with-1-more-feature)
FE and ensemble MLP and LGBM(https://www.kaggle.com/martxelo/fe-and-ensemble-mlp-and-lgbm)
Wavenet with SHIFTED-RFC Proba and CBR(https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr?scriptVersionId=33967419)
There are many others...
Thanks for reading this far.
I'm sorry that my English is bad.

appendix:
```
df = lag_with_pct_change(df, [1, 2, 3])
    
    df['signal_2'] = df['signal'] ** 2

    df['signal_4'] = df['signal'] ** 4 
    
    # calc gradient
    g = df['signal']
    n_grads = 2
    for i in range(n_grads):
        g = np.gradient(g)
        df['grad_' + str(i+1)] = g
        
    # lpf
    x = df['signal']
    n_filts = 5
    wns = np.logspace(-2, -0.3, n_filts)
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        df['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        df['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
    
    # hpf
    x = df['signal']
    n_filts = 5
    wns = np.logspace(-2, -0.1, n_filts)
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        df['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        df['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
```
