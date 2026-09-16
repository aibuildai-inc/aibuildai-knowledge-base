# 23th Place Interesting Approach

Competition: liverpool-ion-switching
Rank: #23
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153799

Hi everyone,
From the beginning of the competition and after reading the paper from the host, I've been thinking about 2 ways to improve the models.
1. Transfer learning from 1f models to 3\5\10 models ==&gt; Failed with HMM\RF\Wavenet...
2. Data augmentation from 1f signals\open_channels to 3\5\10 signals\open_channels: From the intuition that model with 3 max. open channels should be like 3 1f models running with summarized results ==&gt; **Success**

**For the second approach:**
1. Split train\validation set in each fold
2. Random select signals\open_channels sequence (len=4000) from 1f models
3. Randomly picked, for example, 5 sequences from step 2. and sum up to create a sample with 5 max. open channels. 
4. For each summed signal, adjust the mean\variance according to the corresponding mean\variance of the original data.

My wavenet scored around **cv .9388, lb .941**
After **augmenting data for 5\10 open channels**, the wavenet scores **cv .9423, lb: .945**
I've shared my code here: https://www.kaggle.com/khyeh0719/wavenet-with-augmentation-2
which scores around .944 on public lb for a single model with single fold (no RFC features)

Hopefully, this method could be used extensively to create signals\open_channels without actually generating the signal (but assumingly the mean\variance of each open channel state is known in advance :p)

Anyways, I hope you like it and feel it interesting! Thanks Kaggle and the host for such an interesting competition, I really enjoyed it, and finally big congrats to all winners!
