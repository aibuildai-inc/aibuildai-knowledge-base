# 11th Place Solution

Competition: liverpool-ion-switching
Rank: #11
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/154264

Congrats to everyone who participated in this competition. I want to thank Liverpool and Kaggle for putting it on. No competition is perfect but I think that despite it's faults this competition was successful (at least for me) in that it challenged me to think hard and apply machine learning solutions to solve a tough problem. A little over a year and a half ago I participated in my first serious Kaggle competition. I would've never believed that eventually I could achieve a solo gold. It was a lot of work paired with a healthy dose of luck.

## Background (My Competition Story)
This competition piqued my interest the day it was launched. It seemed check all the boxes of what I'm interested in, namely: Non traditional data that wasn't too large or too small, a scientific based problem statement, non-synchronous so I didn't have to depend on kernels for training or submitting, and an opportunity to expand my skills building neural nets. After the Bengali competition ended I started working on the problem on nights and weekends like I normally do and keeping ideas in the back of my mind.

Then COVID-19 cases started to show up in the US and things changed very quickly with my day job, I was furloughed, and suddenly had more time in my schedule for family and kaggle :)

I noticed that many well known kagglers were opting out of this competition, which usually is not a good sign. Some theorized early on that we had reached optimum solution at 0.938 - I was skeptical. Yes, the data is semi-synthetic, but knowing the data was passed through a physical amplifier I figured that signal processing would at least give a boost beyond 0.938.

I started modeling as I usually do with LGBM similar to what I shared in my public kernel (https://www.kaggle.com/robikscube/ion-switching-5kfold-lgbm-tracking) Being inspired by The Zoo's past solutions I wanted to keep my model light on features. After reading @cdeotte 's excellent notebook explaining drift it was clear de-drifting was the priority. My first attempt at removing drift was just fitting a linear and parabolic equation on the rolling median values of each batch and subtracting it from the signal. This gave me a relatively clean signal however there was still one issue, the peaks of each channel were not lined up- specifically the 10 channel signal was not properly lined up with the peaks for the associated channels in other groups. To account for this I manually added an offset that would line up these peaks. This "peak alignment" combined with the clean data my LGBM model achieved 0.941 public LB - good enough for 2nd place at the time. I was thrilled but knew it was still very early in the competition. Not soon after that the first version of the "drift free" data was shared and many teams jumped to 0.941.

[0.941]

I experimented with different features, different CV setups, model settings and was seeing small gains. During this time I also focused on better understanding the macroF1 metric and how it could be optimized. After reading some papers and past kaggle posts I learned that MacroF1 could be improved by giving preference to lower occurring classes. I developed some code for doing this and found post processing gave around a 0.0001 to 0.0005 boost on my CV and LB score but was also quite risky because there is always a chance of over post processing and doing more harm than good.

That's when I decided to take my best LGBM features and use them in a Neural Network. Thanks to the wonderful notebooks I decided to code with TF2/keras instead of pytorch:

- Unet: https://www.kaggle.com/kmat2019/u-net-1d-cnn-with-keras Thanks- @kmat2019
- Wavenet: https://www.kaggle.com/siavrez/wavenet-keras Thanks- @siavrez 

Wavenet showed the most promise, so I started to experiment with different features and slightly modifying the structure which was enough to push me to 0.942 public LB (3rd place at the time).

[0.942]

I experimented with pseudo labeling, target encoding, and many other things but eventually was starting to drop down in the rankings and became very discouraged. I even reached out to a friend/kaggler to see if he was interested in teaming- Luckily (for me) I made my next breakthrough before he could respond.

In one of my experiments I tried was adding an LSTM after the wavenet in combination with target encoding on the signal rounded to 2 decimal places. This gave a big improvement on CV but I was skeptical. When I submitted I was surprised that it held up and I was the first to break 0.945! I spent the next few days trying to understand why this worked and recreating it in a more CV stable way. What I found was that the model performed better when the target encoding wasn't scaled: so while the target encoding was in the range 0-10, the signal and other features were scaled 0-1. I also realized that rounding the signal to 2 digits was essentially the same thing as binning the signal - so I switched to using `KBinsDiscretizer` to allow me to experiment with different bin sizes.

[0.945]

After many experiments I was able to push my score up to 0.946 (1st place at the time). I also found this great kernel gave me the idea to add a 50Hz and a few additional frequency features which gave my CV/LB a boost: https://www.kaggle.com/johnoliverjones/frequency-domain-filtering - thanks @johnoliverjones

[0.946]

By this point many power teams were forming with multiple GMs and strong kagglers. I was preparing myself to be overtaken- but alas I kept experimenting! I tried to keep my GPUs running experiments at all times (`while ps -p 12345; do sleep 60; python myscript.py` is a nifty trick to kick off a script once the previous completes). I also used google colab for additional experiments. Almost all my experiments failed or were inconclusive, but two ended up improving my score to 0.947 (2nd place public):

1) Improving upon my "peak shift" code to find the perfect shift in peaks of each batch.

2) Removing specific frequencies that contained noise (namely 50Hz) from the signal.

[0.947]

And now onto the details:

## Preprocessing
**Removing bad training data**
I started with drift free data as most people did. I also could visually identify that there were some outliers due to extreme noise that didn't resemble anything in the test set, so I removed that section of the training data completely:

```
FILTER_TRAIN = '(time &lt;= 47.6 or time &gt; 48) and (time &lt;= 364 or time &gt; 382.4)'
train = train.query(FILTER_TRAIN)
```
**Removing noisy frequency**
I knew the 50Hz noise was from the hum that is created due to AC current in the amplifier. Having previously studied and worked in power engineering I knew that 50Hz (60Hz in the US) is only the **approximate** frequency of AC power. In reality it can lead or lag 50Hz due to many different factors. I wrote some code to identify where exactly the peak occurred for each batch- some batches it was 50.01Hz others 49.99Hz or 50Hz. The code then automatically tuned a notch filter with a Q value tuned with scipy to minimize the variance of the ~50Hz frequency with the surrounding frequencies amplitudes. The allowed it to "smoothly" filter out the 50Hz frequency with minimal disturbance to the rest of the signal.

This notebook was helpful when I was first researching the idea: https://www.kaggle.com/kakoimasataka/remove-pick-up-electric-noise

And the code for this "auto 50Hz filter":

```
def fix_50hz(tt, batch, Q = 250.0,
             f0 = 50, fs=10000, plot=True,
             write=True, domax=True,
             use_cleaned=False):
    """
    Q: Quality factor
    f0: Frequency to be removed from signal (Hz)
    """
    if use_cleaned:
        sig_sample = tt.loc[tt['sbatch'] == batch]['signal_cleaned_10s'].values
    else:
        sig_sample = tt.loc[tt['sbatch'] == batch]['signal'].values
    
    tmin = tt.loc[tt['sbatch'] == batch]['time'].min()
    tmax = tt.loc[tt['sbatch'] == batch]['time'].max()
    chunique = tt.loc[tt['sbatch'] == batch]['open_channels'].nunique()
    
    fft = np.fft.fft(sig_sample)
    psd = np.abs(fft) ** 2
    fftfreq = np.fft.fftfreq(len(psd),1/fs)
    
    # See if 50 hz is the max amplitude
    i_45_55 = (fftfreq &gt;= (f0-5)) &amp; (fftfreq &lt;= (f0+5))
    i_45_55_no_50 = ((fftfreq &gt;= (f0-5)) &amp; (fftfreq &lt;= (f0+5))) &amp; (fftfreq != f0)
    i_50 = fftfreq == 50
    if domax:
        freq_with_max_ampl = fftfreq[i_45_55][np.argmax(psd[i_45_55])]
        if freq_with_max_ampl &gt; (f0-1) and freq_with_max_ampl &lt; (f0+1) and freq_with_max_ampl != f0:
            print(f'Making f0 {freq_with_max_ampl}')
            f0 = freq_with_max_ampl
    else:
        freq_with_max_ampl = fftfreq[i_45_55][np.argmin(psd[i_45_55])]
        if freq_with_max_ampl &gt; (f0-1) and freq_with_max_ampl &lt; (f0+1) and freq_with_max_ampl != f0:
            print(f'Making f0 {freq_with_max_ampl}')
            f0 = freq_with_max_ampl

    # Design notch filter to make frequency f0 similar to average
    def notch_filter(Q):
        b, a = signal.iirnotch(f0, Q, fs)
        # Filter and plot
        y = signal.filtfilt(b, a, sig_sample)

        fft_ = np.fft.fft(y)
        psd_ = np.abs(fft_) ** 2
        fftfreq_ = np.fft.fftfreq(len(psd_),1/fs)
        ivar = ((fftfreq &gt;= f0-2) &amp; (fftfreq &lt; f0-1)) | ((fftfreq &gt;= f0+2) &amp; (fftfreq &lt; f0+1))
        i_f0 = fftfreq_ == f0
        abs_diff = np.abs(np.mean(psd[ivar]) - psd_[i_f0])
        diff_others = np.mean(psd[ivar] - psd_[ivar])
        return abs_diff + np.abs(diff_others / 2)
    
    res = minimize(notch_filter, [Q], tol=1e-10, method='Powell') #, bounds=((100, 5000),))
    Q = res['x']
    print(f'Using Q {Q}')
    b, a = signal.iirnotch(f0, Q, fs)
    # Filter and plot
    y = signal.filtfilt(b, a, sig_sample)

    fft_ = np.fft.fft(y)
    psd_ = np.abs(fft_) ** 2
    fftfreq_ = np.fft.fftfreq(len(psd_),1/fs)
    if plot:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 2))

        i = abs(fftfreq) &lt; 100
        ax.grid()
        ax.plot(fftfreq[i], 20*np.log10(psd[i]), linewidth=1)
        ax.plot(fftfreq_[i], 20*np.log10(psd_[i]), linewidth=1)
        ax.set_title(f'Time: {tmin} to {tmax} - Unique Channels: {chunique} - Q: {Q}')
        ax.set_xlabel('Frequency (Hz)') 
        ax.set_ylabel('PSD (dB)')
        ax.axvline(f0, color='red', alpha=0.2)
        ax.axvline(-f0, color='red', alpha=0.2)
        plt.show()
    if write:
        # Write Cleaned Signal to dataset
        tt.loc[tt['sbatch'] == batch, 'signal_cleaned_10s'] = y
    return tt
```

This worked rather nicely at removing the 49-51Hz noise. I experimented running this on each batch individually, by 10 second intervals, and 1 second intervals. I also experimented with removing other "noisy" frequencies. Plots were used to validate the change (the blue area shows where the frequency was removed)

[Filter Example]

**Peak Alignment**
The final step of preprocessing was lining up of peaks of the distribution for each batch. To achieve this I wrote a bit of code that minimized the difference in median signal between batches by `open_channel`. In doing some research I found that these types of "peak alignment" problems are quite common in various field and I skimmed a few papers about it. [For instance this paper discusses peak alignment for chromatography](https://www.researchgate.net/publication/221816197_Model-based_peak_alignment_of_metabolomic_profiling_from_comprehensive_two-dimensional_gas_chromatography_mass_spectrometry)

And the code:

```
def apply_shift(tt, shift_group, base_groups):
    
    te1 = tt.query('sbatch == @shift_group')['time'].min()
    te2 = tt.query('sbatch == @shift_group')['time'].max()
    
    print(f'Running for Group {shift_group} - {te1} to {te2}')
    shift_channel_freq = tt.query('sbatch == @shift_group')['open_channels'].value_counts().sort_values(ascending=False).index
    base_channels = tt.query('sbatch in @base_groups')['open_channels'].unique()
    for m in shift_channel_freq:
        if m in base_channels:
            print(f'Lining up on channel {m}')
            most_freq_channel = m
            break
            
    m1 = tt.query('sbatch in @base_groups and open_channels == @most_freq_channel')['signal_fixed'].median()
    m2 = tt.query('sbatch == @shift_group and open_channels == @most_freq_channel')['signal'].median()
    shift_start = m1 - m2

    print('Base Shift: ', shift_start)

    def overlap_weighted(shift, shift_group=shift_group, base_groups=base_groups, tt=tt):

        shift_data = tt.query('sbatch == @shift_group').copy()
        shift_data['signal_fixed'] = shift_data['signal'] + shift
        base_data = tt.query('sbatch in @base_groups').copy()
        shift_times = shift_data['time'].values

        common_channels = np.intersect1d(base_data['open_channels'].unique(),
                   shift_data['open_channels'].unique())

        base_data = base_data.query('open_channels in @common_channels')
        shift_data = shift_data.query('open_channels in @common_channels')
        shift_medians = shift_data.groupby('open_channels')['signal_fixed'].median()
        base_medians = base_data.groupby('open_channels')['signal_fixed'].median()
        shift_counts = shift_data.groupby('open_channels')['signal_fixed'].count()
        base_counts = base_data.groupby('open_channels')['signal_fixed'].count()
        weighted_diff = ((shift_medians - base_medians) * shift_counts).mean()  / shift_counts.sum()
        print(f'Shift: {shift[0]:0.6f} : Weighted Diff {weighted_diff:0.20f}', end="\r", flush=True)
        return np.abs(weighted_diff)
    print('')
    res = minimize(overlap_weighted, shift_start, method='Powell', tol=1e-2)
    best_shift = res['x']
    print('')
    print('best_shift', best_shift)
    
    te1 = tt.query('sbatch == @shift_group')['time'].min()
    te2 = tt.query('sbatch == @shift_group')['time'].max()

    tt.loc[(tt['time'] &gt;= te1) &amp; (tt['time'] &lt;= te2), 'signal_fixed'] = \
        tt.loc[(tt['time'] &gt;= te1) &amp; (tt['time'] &lt;= te2)]['signal'] + best_shift

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(15, 3))
    tt.query('sbatch == @shift_group')['signal_fixed'].plot(kind='kde', ax=ax, title='Shift Set')
['signal'].median().iteritems():
@shift_group').groupby('open_channels')['signal_fixed'].median().iteritems():
        ax.axvline(i[1], color='orange', ls='--')
    tt.query('sbatch in @base_groups')['signal_fixed'].plot(kind='kde', ax=ax, title='Base set')
    for i in tt.query('sbatch in @base_groups').groupby('open_channels')['signal_fixed'].median().iteritems():
        ax.axvline(i[1], color='red', ls='--')
    plt.show()
    return tt

```
Here are two examples of the output from this code:

[Peak 1]

[Peak 2]

The one issue with the peak alignment process was that for the training set I knew the true open channels. For the test set I didn't know them, so I used my previous model's predictions. I tried using out of fold values on the training set- but it didn't work as well. My thinking was that eventually my test alignment would reach an "optimum" shift.

Still, I knew this left me open to the possibility of overfitting to my own predictions. I tried many other approaches for lining up the peaks- but decided that this seemed to work the best despite the risks.

The final result was a very clean dataset with peaks of each channel aligned.

[Clean 1]
[Clean 2]

## Features
As I mentioned above the most important feature I found was target encoding. I used the `KBinsDiscretizer` to first divide the signal into different bins and then ran target encoding on that. I didn't scale the target encoding but did scale all other features. Other features were lags +/- 5 samples, signal^2 and the 50Hz features.

The following is the function I used to create target encoding. It makes use of the category-encoders package https://pypi.org/project/category-encoders/

```
def add_target_encoding(tr_df, val_df, test_df,
                        features, n_bins=500,
                        strategy='uniform', feature='signal',
                        smoothing=1):
    if smoothing == 1:
        sm = ''
    else:
        sm = '_' + str(smoothing)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    tr_df[f'{feature}_{n_bins}bins{sm}'] = kbd.fit_transform(tr_df[feature].values.reshape(-1, 1))
    val_df[f'{feature}_{n_bins}bins{sm}'] = kbd.transform(val_df[feature].values.reshape(-1, 1))
    test_df[f'{feature}_{n_bins}bins{sm}'] = kbd.transform(test_df[feature].values.reshape(-1, 1))

    tr_df[f'{feature}_{n_bins}bins{sm}']  = tr_df[f'{feature}_{n_bins}bins{sm}'].astype('category')
    val_df[f'{feature}_{n_bins}bins{sm}']  = val_df[f'{feature}_{n_bins}bins{sm}'].astype('category')
    test_df[f'{feature}_{n_bins}bins{sm}']  = test_df[f'{feature}_{n_bins}bins{sm}'].astype('category')

    te = ce.target_encoder.TargetEncoder(verbose=1, smoothing=smoothing)
    tr_df[f'target_encode_{feature}_{n_bins}bins{sm}'] = te.fit_transform(tr_df[f'{feature}_{n_bins}bins{sm}'],
                                                                            tr_df['open_channels']) 
    val_df[f'target_encode_{feature}_{n_bins}bins{sm}'] = te.transform(val_df[f'{feature}_{n_bins}bins{sm}']) 
    test_df[f'target_encode_{feature}_{n_bins}bins{sm}'] = te.transform(test_df[f'{feature}_{n_bins}bins{sm}'])

    te_col_name = f'target_encode_{feature}_{n_bins}bins{sm}'

    if te_col_name not in features:
        features.append(te_col_name)

    return tr_df, val_df, test_df, features
```

Every so often I would permutation importance on my models to see if any features could be dropped.

An example of the exact features used in one of my final models is:
	
```
['signal', 'f_DC', 'f_50Hz', 'f_n100Hz',
'f_2xn100', 'signal_mm_scaled', 'signal_round2',
'signal_shift_pos_1', 'signal_shift_neg_1',
'signal_shift_pos_2', 'signal_shift_neg_2',
'signal_shift_pos_3', 'signal_shift_neg_3',
'signal_shift_pos_4', 'signal_shift_neg_4',
'signal_shift_pos_5', 'signal_shift_neg_5',
'signal_2',
'target_encode_signal_round2_500bins_10',
'target_encode_signal_300bins',
'target_encode_signal_600bins',
'target_encode_signal_100bins',
'target_encode_signal_1000bins']
```

## Model: Wavenet+LSTM

My model ended up being based off of the one found in the wavenet-keras kernel https://www.kaggle.com/siavrez/wavenet-keras (thanks @siavrez). I wanted to upvote his notebook so badly during the competition but I thought it might give people a clue as to what I was using. I've since upvoted it and wish I could upvote it 10x it was so helpful.

Building off of the wavenet from that notebook I added:

- LSTM right after the wavenet into a softmax output
- Added GaussianNoise seemed to help
- Tuned learning rate scheduler

Things that didn't work:

- GRU wasn't as good as LSTM
- GRU + LSTM and concat results
- Change activation functions
- Different levels of GaussianNoise
- 32, 64, 92, 128 sized LSTM (92 seemed to work best)
- Lots of different settings to the wavenet but nothing conclusively improved.
- Lots of different learning rate setups
- Modeling types seperately
- Removing training data in attempt to more closely resemble public or private test data.

I tried many, many different ideas, an example of one my final models looked like this:

```
    inp = Input(shape = (shape_))
    x = wave_block(inp, 16, 3, 12)
    x = GaussianNoise(0.01)(x)
    x = wave_block(x, 32, 3, 8)
    x = GaussianNoise(0.01)(x)
    x = wave_block(x, 64, 3, 4)
    x = GaussianNoise(0.01)(x)
    x = wave_block(x, 128, 3, 1)
    x = LSTM(96, return_sequences=True)(x)
    out = Dense(11, activation = 'softmax', dtype='float32', name='predictions')(x)
    model = models.Model(inputs = inp, outputs = out)

```

## Augmentation
Early on I found that flipping the signal during training to double the data size helped. I also flipped for TTA (used flipping on the test data when predicting and averaged the results). I tried adding noise to the raw signal but that only made CV and LB worse.

## The final week- training and blending
There were a lot of things that were difficult about doing a competition solo - but this part was probably the hardest. I didn't have anyone to consult with about how to make my final ensemble and blend and had to rely on my gut, in the end it ended up working out - but I also had a bit of luck.

Since I was worried about already being slightly overfit in my peak alignment I ended up taking my best 40 or so submissions and blending the raw probabilities before doing post processing. For one of my final submissions I just took a straight average across the 40 models. The second submission I weighed each prediction by it's position on the public LB. I also very closely inspected the correlation of each sub, comparing public vs private LB, and also comparing by model type. My main concern was with the 10 channel section in the private test set. I noticed that was the section that had the highest variance across submissions, and it would have a huge impact on the final score because each class is weighed equally with the macroF1 metric.

Example plot correlation of submission files ordered by public LB score:
[Corr]

## Post Processing

For post processing I merged two ideas from other kagglers. The first was from Chris' solution writeup about[ "hacking macro recall" in the bengali competition](https://www.kaggle.com/c/bengaliai-cv19/discussion/136021). The second was the f1 optimization class being used in notebooks. What makes this code unique is that it optimizes using the raw probabilities for each class and not the class predictions themselves.

The resulting code was this:

```
class ArgmaxF1Optimizer(object):
    """
    Class for optimizing predictions using argmax
    """
    def __init__(self, initial_exp=-1.2, method='nelder-mead',
                verbose=True):
        self.initial_exp = initial_exp
        self.opt_result = None
        self.method = method
        self.verbose = verbose

    def apply_pp(self, X, EXP):
        preds_argmax = np.argmax(X,axis=1)
        s = pd.Series(preds_argmax)
        vc = s.value_counts().sort_index()
        df = pd.DataFrame({'a':np.arange(11),'b':np.ones(11)})
        df.b = df.a.map(vc)
        df.fillna(df.b.min(),inplace=True)
        mat1 = np.diag(df.b.astype('float32')**EXP)
        return np.argmax(X.dot(mat1), axis=1)

    def apply_pp_f1_score(self, EXP, X, y):
        preds_pp = self.apply_pp(X, EXP)
        score = -macro_f1_score(y, preds_pp)
        if self.verbose:
            print(f'Exp {EXP}, score {score}')
        return score
    
    def fit(self, X, y):
        loss_partial = partial(self.apply_pp_f1_score, X=X, y=y)
        if self.method == 'nelder-mead':
            self.opt_result = minimize(loss_partial,
                           self.initial_exp,
                           method='nelder-mead')
            self.opt_exp = self.opt_result['x']

        elif self.method == 'brute':
            rranges = ([slice(-1, 1, 0.05)])
            self.opt_result = brute(loss_partial, rranges,
                       full_output=True,
                       finish=fmin)
            self.opt_exp = self.opt_result[0][0]
        if self.verbose:
            print(f'Optimal exp value {self.opt_exp}')
            score_argmax =  macro_f1_score(y, np.argmax(X, axis=1))
            score_pp = - self.apply_pp_f1_score(self.opt_exp, X, y)
            print(f'Score raw:            {score_argmax:0.5f}')
            print(f'Score post processed: {score_pp:0.5f}')
            self.score_argmax = score_argmax
            self.score_pp = score_pp

    def predict(self, X, y=None):
        if self.opt_exp is None:
            print('[ERROR] Must first run fit method')
            return
        return self.apply_pp(X, self.opt_exp)
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
```

I wanted to be extremely careful when applying post processing. I knew it helped on the public leaderboard but could never be 100% confident with how much to apply to the private. To help me decide how much post processing to apply I wrote code that:

- Used the OOF predictions from all of the models in the blend.
- Simulated the exact same distributions of that in the public and private test set (matching both model type and open channel)
- Repeated this process 100x with different seeds. The reason why I did this is because the F1 score would fluctuate +/- 0.001 depending on which samples were selected.
- Ran my F1 optimization code on each simulation.
- Took the average `EXP` from the 100 synthetic cases and used that to post process my final submission.
- Note this was done separately on public and private test set as they are scored separately. 

The output of this simulation looked something like this:

```
0 Raw public:  0.94290 : -0.14906 : 0.94304 	 Raw private: 0.94278 : -0.14000 : 0.94289
1 Raw public:  0.94391 : -0.01950 : 0.94390 	 Raw private: 0.94377 : -0.02000 : 0.94374
2 Raw public:  0.94353 : -0.13569 : 0.94378 	 Raw private: 0.94325 : -0.12837 : 0.94349
3 Raw public:  0.94336 : -0.02237 : 0.94338 	 Raw private: 0.94341 : -0.02000 : 0.94344
4 Raw public:  0.94361 : -0.10450 : 0.94381 	 Raw private: 0.94375 : -0.09656 : 0.94391
5 Raw public:  0.94441 : -0.09394 : 0.94453 	 Raw private: 0.94439 : -0.04750 : 0.94449
6 Raw public:  0.94412 : -0.10000 : 0.94427 	 Raw private: 0.94413 : -0.02200 : 0.94429
7 Raw public:  0.94413 : -0.06891 : 0.94418 	 Raw private: 0.94435 : -0.04828 : 0.94440
8 Raw public:  0.94384 : -0.13325 : 0.94410 	 Raw private: 0.94374 : -0.12375 : 0.94397
9 Raw public:  0.94363 : -0.01900 : 0.94365 	 Raw private: 0.94342 : -0.04100 : 0.94344
10 Raw public:  0.94333 : -0.04812 : 0.94341 	 Raw private: 0.94324 : -0.04550 : 0.94329
11 Raw public:  0.94339 : -0.09656 : 0.94354 	 Raw private: 0.94356 : -0.08803 : 0.94373
12 Raw public:  0.94378 : -0.09000 : 0.94385 	 Raw private: 0.94380 : -0.04200 : 0.94382
13 Raw public:  0.94385 : -0.09112 : 0.94403 	 Raw private: 0.94365 : -0.04750 : 0.94380
14 Raw public:  0.94380 : -0.09562 : 0.94403 	 Raw private: 0.94394 : -0.08775 : 0.94417
....
```
 
The distribution then looked like this, and I used the mean value when applying post processing to my submission:

[exp selection]


## Things I still can't explain.

There are still some lingering questions I have.

- **What are these anchor points?** [-2.5002, -1.2502, -0.0002, 1.2498, 2.4998, 3.7498]. I noticed in the signal there is a spike in unique value counts just to the right of the signal distribution for each open channel. You can see them clearly in this plot. I called them "anchor points" because they almost always were labeled as the open_channel even though they were to the right of the mean value.

[spike]

I probably spent 2 or 3 days working on this and thought it might be helpful for lining up peaks. I still don't have a good explanation for why they exist. I even wrote an optimization function that shifted the drift batches such that when rounding to 4 decimal places it would maximize these values. It resulted in a worse LB score.

- **Where do new batches start?** 

Batch sizes are 10 seconds or 50 seconds.. I think. I considered seconds 0-50 to be one batch until closer inspection of the rolling median value. You can see a clear shift in the median value at 10 seconds. I found the same thing in the private test data. Because I was unsure I decided to split at these points and did my "peak alignment" on these separately.

[shift]

## Conclusion

I ended up shaking down 9 spots in the private leaderboard to 11th place. My best single model scored 0.94547 private and had the potential for 7th place - but there was no way I would've selected it. On the other hand my best public LB score model had a private LB score of 0.94211! I never considered selecting it either.

[selectedsub]

I'm sure there are things I could've done to perform better in the shakeup but I was just happy that it was good enough to stay in the gold range.

Depending on time and interest - I may create a notebook that runs my code end-to-end.

That's it. I hope you found this writeup helpful!
