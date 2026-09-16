# 22nd Place Overview

Competition: talkingdata-adtracking-fraud-detection
Rank: #22
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56571

Thanks to Kaggle &amp; TalkingData for another entertaining diversion :) Congrats to all winners and thanks to all who shared Kernels, tips and solution write-ups, they’re great to read.

This is a really interesting data set and as I related in my [story here][1], click fraud is a really insidious problem that drives people to do really [odd things][2] (thanks @yifanxie - amazing to actually see it!)

## Overview

I used broadly similar features to everyone else, next click times and count features, and stuck to the route recommended by early leaders – using lightgbm and training on the full training set – it surprised me that my 32Gb machine could. After reading the great [shared tips][3], switching to creating a NumPy array upfront then populating that by loading feature columns was very efficient, with 33 features it barely even needed any swap space.

I validated on the 2nd half of the last day of training for simplicity, just one validation set, then I’d save predictions in a dataframe and do groupby(‘hour’) and check AUC that way. I did pay more attention to the hours that matched the test set hours, but I thought other hours might lead to useful insights (but nothing noteworthy to report).

My general process for one iteration was quite lightweight:

1. an hour of feature engineering – save columns individually
2. a 1.5-2 hour validation run
3. a 3 hour full model build

After setting this up, the only real manual effort is step 1, the validation run just determines # of trees (which was quite stable over feature sets, but sometimes repeated if the results were poor). The full model build is a one click process that builds a model, makes test set predictions with varying numbers of trees (to test assumptions about increased data set size &amp; scaling # of trees), then uploads the submissions via the highly recommended [Kaggle API][4].

I managed 13 iterations, never more than one in a day, so it was a bit of a slow waiting game, and a long way from optimal.  Getting results in fewer iterations is down to some intuition and luck in selecting what works. There are a lot of features I tried that didn’t work, so my remaining features are mostly fairly baseline/simple things… ([Matrix factorization of IP vs app log click counts][5] is the one thing I really wish I’d tried, it will summarize an IP’s affinity for apps and interact very well with the app and IP count features… Well done to @cpmpml &amp; other top teams :)

## Entropy features

Whereas most people used nunique, I thought that might be noisy. With a click count histogram like e.g. counting os’s for an app = [12345, 5, 1, 1] it’s technically 4 unique os’s, but very low entropy (it’s more like one os).

The original feature encodings made this easier: they were all 0..n, so it was simple to do it all in NumPy using `np.add.at()` to fill a 2D array with counts, then summarize like this:

    from scipy.stats import entropy
    
    # e.g. a=’ip’ ; b=’app’
    cc = np.zeros((int(df[a].max()+1), int(df[b].max()+1)), dtype=np.int32)
    np.add.at(cc, (df[a].values, df[b].values), 1)
    
    (cc&gt;0).sum(0) # this is nunique for a
    (cc&gt;0).sum(1) # this is nunique for b
    map(entropy, cc)    # entropy of a over b
    map(entropy, cc.T)  # entropy of b over a

All two way interactions were possible with 32Gb of RAM, and took only minutes to compute this way. I also combined device &amp; os into one new categorical (only 6561 unique values), and app &amp; channel (1518 unique values), which enabled some 3-way interactions like entropy of IP over (device, os).

## Sessions

Four new features:

 - sort by ip, dev, os
 - define max gap e.g. 60 seconds
 - every time there is a change in ip/device/os, or a gap in the clicks, assign categorical session ID
 - for sessions, summarize:
    - duration
    - count
    - click rate (duration/count)
    - app entropy

I intended to add entropy over gaps, e.g. lots of similar gaps of 5-6 secs = low entropy = bots. Thought maybe click rate was enough.

## Naive Count Features

Normally, count features group on one or more columns and count the actual occurrences. Inspired by Naive Bayes classifiers: another view on the data is to assume the fields are independent, and do:

 - univariate count
 - divide counts by the number of rows to get a probability of seeing the value
 - log() the probability
 - sum different combinations of these columns similarly to normal feature interactions (i.e. -, +, /, *).

Mathematically, this is the same as a geometric mean probability of seeing the record, under the assumption the columns are independent. Again, extremely fast to compute using NumPy alone.

The combinations &amp; loading code I used:


    def loadraw(name, dtype):
        return np.fromfile(name, dtype=dtype)
    
    def log_p_feat(t, col):
        # t is ‘train’ or ‘test’
        return loadraw('../feats/%s_log_p_%s'%(t,col), np.float32)
    
    x[:,a] = log_p_feat(t, 'ip') + log_p_feat(t, 'dev') + log_p_feat(t, 'os')
    x[:,a+1] = x[:,a] + log_p_feat(t, 'app')
    x[:,a+2] = x[:,a] + log_p_feat(t, 'chan')
    x[:,a+3] = x[:,a] + log_p_feat(t, 'app') + log_p_feat(t, 'chan')


Adding these features in with normal 2D/3D counts might pick up on interesting things. With these ‘naive’ counts: rare values overlap - a low (log) probability row might be rare IP with common OS ***or*** a rare OS with common IP. (Although IP was much higher cardinality, so probably dominates the log sum. A weighted mix might be better.)

This can produce a very large number of unique values, so lightgbm value binning might come to the rescue here and avoid over-fitting. I’m not 100% convinced this was a good idea… but it did lead to a good single model (0.9811 public, 0.9825 private) where these naive count features had high feature importances.

## Bit Shifting

Small tip: by looking at the maximum values for each column, you can work out how many bits are required to store each, and happily, the five categoricals all fit in a single 64 bit integer. You can pack them like this:

    def do_pack(df):
        a = 0
        a += df.app.values.astype(np.uint64)
        a += df.os.values.astype(np.uint64)&lt;&lt;10
        a += df.channel.values.astype(np.uint64)&lt;&lt;20
        a += df.ip.values.astype(np.uint64)&lt;&lt;30
        a += df.device.values.astype(np.uint64)&lt;&lt;50
        return a

They can be unpacked (these work on values or NumPy arrays):

    def fapp(v): return v&amp;0x3ff
    def fos(v): return (v&gt;&gt;10)&amp;0x3ff
    def fchan(v): return (v&gt;&gt;20)&amp;0x3ff
    def fip(v): return (v&gt;&gt;30)&amp;0xfffff
    def fdev(v): return (v&gt;&gt;50)&amp;0xfffff

This is vectorized, so runs extremely quickly, and the result is like a hash, but without collisions. You can use the result to do groupby() operations - more efficiently in both time &amp; RAM. You can also ignore fields by setting all their bits to one, e.g. to ignore device `a |= 0xfffffl&lt;&lt;50`. (Because 0 was a valid value for each field.) Using `np.save()` (or `a.tofile` and `np.fromfile`) on the 64 bit array you could save the entire 4 days of train &amp; full test into 1.9Gb on disk. Times could be similarly packed into 32 bits, about 900Mb… With these copies of the data I actually managed to do some feature exploration in idle moments (in Java) on an old 8Gb MacBook!

## Categoricals

Grouping on all five combined fields, there were many different series, some with 10k-20k appearances, but a long tail of shorter series. Instead of separate count &amp; cumcount features I tried a categorical feature to mark where the row is in the rarer/shorter series. Using max_bin of 255, there are 8 bits to play with: 4 bits to mark the length of the series (1..15) and 4 bits to mark where in the series the record appeared (0..14).

Where v is all five fields combined as above, use groupby(v).size() to get df.series_size and groupby(v).cumcount() to get df.series_cumcount, then:

    idx = df.series_size.values&lt;=15
    cat = np.where(idx, df.series_size.values&lt;&lt;4, 0)
    cat |= np.where(idx, df.series_cumcount.values, 0)

So this is one feature that encodes the length of the ‘series’ of [ip,dev,os,app,chan] **and** the position within the series. Tree splits can then address subsets like:

    if cat == 0x32
    if 3_element_series and this_record_is_last_in_series
    
    if cat == 0x20
    if 2_element_series and this_record_is_first_in_series

To make the latest versions of lightgbm do these kind of splits you need to set the `max_cat_to_onehot` parameter correctly, in this case to 256 or more – but beware this applies to all categorical features, so for example channel may be used for one hot style splits too – it depends on how many values are seen in each column at dataset construction time (lightgbm samples data to construct the bins). All model parameters/regularizations are trade-offs of some kind.

Limited time and iterations mean I can’t be sure this didn’t overfit, particularly for series that span train &amp; test times, and things like “length 14 series, element 4” which is not likely to mean anything (the hope is, this could uncover some really quirky leak!) Perhaps addressing only shorter series would be better, or chunking the series by time. I relied on the model to tell me which were more important - a quick inspection of models shows low values like 0x20, 0x21 were used the most.

## Leaderboard

From https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56524

![JT public LB][6]

For me, validation scores stayed very consistent, but leaderboard scores were mysteriously lower. It was only eventually removing some features that fixed this: all of the count features that used the corrected Chinese timezone day. (These were: `ip_day_count`, `app_channel_day_count`, `device_os_day_count`.) I’m still not sure what caused that, and it’s puzzling that other’s made these work, but removing them on the penultimate day finally achieved validation / leaderboard correlation, getting me to public #105.

As #105 slid down into the #200’s (I need finer resolution on that graph!) I gave in and as a backup plan rank averaged my main sub with Dirk’s which was worth 2e-5 of AUC and one spot on the private LB!


                  private      public    model
    single lgbm 0.9825639    0.9811706   v12
    single lgbm 0.9825211    0.9810671   v13
    selection 1 0.9828068    0.9813324   rank average v12+v13 (Private #23)
    selection 2 0.9828279    0.9816611   rank average v12+v13 + Dirk (Private #22)


  [1]: https://www.kaggle.com/jtrotman/eda-talkingdata-temporal-click-count-plots/code#310213
  [2]: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/54765
  [3]: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55325
  [4]: https://github.com/Kaggle/kaggle-api
  [5]: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283
  [6]: https://kaggle2.blob.core.windows.net/forum-message-attachments/326995/9398/jt_public_lb.png
