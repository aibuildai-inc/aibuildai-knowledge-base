# 13th Place Solution

Competition: g2net-gravitational-wave-detection
Rank: #13
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275841

I would like to thank my team members @sakshamaggarwal @darkravager @callmeb and @mrigendraagrawal, organizers and the Kaggle community for the amazing experience we had during this competition.

# Preprocessing: 
thanks to @callmeb
 - **Mean PSD**: We calculated the mean PSD as simple running mean and store it for later normalization:
```
train0 = train[train.target==0].reset_index(drop=True)
DET = None
for idx, (id, target, path) in tqdm(train0.iterrows(), total=len(train0)):
    ts = np.load(path)
    ts = ts_window(ts)
    ts = ts_whiten(ts)
    ts = torch.tensor(ts)
    fs = torch.fft.fft(ts)
    if DET == None:
        DET = fs.abs()
    else:
        DET = (fs.abs() + idx * DET) / (idx+1)
```
 - **Data whitening** : you can use any whitening you want. This is the one of many solutions we used.
```
WINDOW=signal.tukey(4096, 1/4)[None,:]
def ts_window(ts):
    return ts * WINDOW
def ts_whiten(ts, lf=24, hf=364, order=4):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=2048)
    normalization = np.sqrt((hf - lf) / (2048 / 2))
    return signal.sosfiltfilt(sos, ts) / normalization
```
 - After loading the data we **Normalize** the signal as follows:
```
ts = ts_window(ts)
ts = ts_whiten(ts)
ts = torch.fft.ifft(fs).real
fs = fs / DET
ts = torch.fft.ifft(fs).real
(you can also reuse the window here)
```
# Augmentations
 - Random channel shuffle
 - Random rolling (time shift)
# Approaches
Used stratified 5-fold training strategy. We had some ideas which gave a little boost to its native counterparts. The approaches were:
 - **3 Channel (native)** - stacked the detectors channel wise. 
 - **6 Permutations** -  We horizontally stacked the 3 detectors with different permutations. Then stacked these 6 permutations channel wise.
 - **CQT+CWT** - We took 3 channels(for 3 detectors) from CQT transformation and 3 from CWT transformation and stacked them.
 - **Double CWT** - created two parallel CWT transformations with different parameters, with the idea of better frequency resolution by one and better time resolution by another.

**NOTE** : **Parallel** keyword denotes a modified architecture where we trained two CNN backbones simultaneously and concatenated their GAP’s output followed by two FC layers.

We trained backbones of different complexities which may use a specific approach or a combination of them. The models also varied with image sizes and frequency range. 
[model list]
# Ensembing
 - We had around 50 models at the end with our best model scoring 0.8808.
 - With weighted averaging, we were able to achieve around 0.8823 public LB ( 0.88047 private).
 - Switching to Stacking Ensembling with sklearn’s MLP classifier significantly boosted our public LB to 0.8830 ( 0.88133 private).
 - Also tried stacking with different classifiers like LGB, logistic etc and different model selections like top 10, top 20 etc, but they performed relatively poorer in both public and private LB.

Special thanks to [Solvers Club](https://www.solversclub.com/) for the generous GPU resources they provided. Alone Kaggle resources would not have have suffice for the competition (forever queued TPUs)

Although the shakeup was little, it was enough to bring us down from gold to silver. 
Hope had trained 1D CNNs :)
