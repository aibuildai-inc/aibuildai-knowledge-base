# 1st Place Solution [PND]

Competition: prostate-cancer-grade-assessment
Rank: #1
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169143

Congratulations to everyone and thanks for the hosts for preparing this competition!

#### I published [slide](https://docs.google.com/presentation/d/1Ies4vnyVtW5U3XNDr_fom43ZJDIodu1SV6DSK8di6fs/edit?usp=sharing)!
#### Our code is [here](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)!

# Proposed Denoising Method
We're very suprised that we finished 1st, and our simple label-denoising method (suprisingly) boosted up PB.

The competition was all about handling noisy labels, so we worked hard on finding good ways to denoising.

Here is our simple denoising method by @kyoshioka47:

## Getting cleaned labels
- train k-folds with effnet-b1 (Almost identical to Qishen's kernel)  
  - Model specifics in fam_taro( @yukkyo ) part
- Predict hold-out sets with the trained model. We get `pred` with this step.
- Remove the training data which has a high disparity between ground truth and pred. The filtered labels will be called cleaned labels.

We calculate `disparity` by the absolute difference of ISUP between GT and pred. Data with disparity larger than 1.6 was simply removed.  

Here is the psuedo codes. probs_raw is the raw prediciton results (ISUP)

```python
# Base arutema method
def remove_noisy(df, thresh):
    gap = np.abs(df["isup_grade"] - df["probs_raw"])
    df_removed = df[gap &gt; thresh].reset_index(drop=True)
    df_keep = df[gap &lt;= thresh].reset_index(drop=True)
    return df_keep, df_removed

df_keep, df_remove = remove_noisy(df, thresh=1.6)
show_keep_remove(df, df_keep, df_remove)
```




## Retraining
Retrain model with using the denoised labels. 
We get CV 0.94 LB 0.90 PB 0.934 with a simple Qishen Eff-b0 model with k-folds.
Ensambling with different models further boosted to 1st place.

We tried CleanLab too, but that did not perform well in CV/LB so we sticked with this.

# 1. Our final submission

- Select 1 (public LB 0.910, private LB 0.922)
    - Resnext50_32x4d(poteman)
- Select 2 (public LB 0.904, private LB 0.940)
    - Effnet-B0(arutema47) + Effnet-B1(fam_taro)
        - Simple average (`1 : 1`)

Suprisingly, even with several weight patterns, the PB was 0.940.

# 2. Resnext50_32x4d( @poteman ), public 0.910, private 0.922
This was our best LB model.

- Split kfold: stratified kfold with imghash(threshold 0.90)
- iafoss tile method
    - tile size 256, tile num 64
- model:resnext50_32x4d
- head: 3 * reg_head + 1 * softmax head

# 3. Effnet-B1(fam_taro), public 0.901, private 0.932

- Split kfold
    - stratified 5 kfold with gleason-score and imghash similarity (threshold 0.90)
        - convert `negative` to `0+0`
        - how to grouping by imghash similarity
            - This is based on @appian 's kernel
                - https://www.kaggle.com/appian/panda-imagehash-to-detect-duplicate-images
            - https://www.kaggle.com/yukkyo/imagehash-to-detect-duplicate-images-and-grouping
    - In my opinion, split method is import point for our denoise method.
        - Because we use prediction of out of fold
        - If you put the duplicate images in a different fold, I don't think denoise will work for them
- Data
  - iafoss tile method
    - tile size 192, tile num 64
- Model: Effnet-B1 + GeM
    - label: isup-grade and first score of gleason(10 dim bin)
- Make final sub by 3 steps
    - Local train &amp; predict
    - Remove noisy label
        - extended @kyoshioka47 method
        - Change threshold for each isup-grade and data-provider
    - Re-train
- Not work for me
    - Remove noisy by confident-learning
    - Cycle GAN augmentation(karolinska &lt;-&gt; radboud)
    - test with AdaBN &amp; Freezing BN at train
    - CutMix, Mixup (before denoising)
  
```python
def remove_noisy2(df, thresholds):
    gap = np.abs(df["isup_grade"] - df["probs_raw"])
    
    df_keeps = list()
    df_removes = list()
    
    for label, thresh in enumerate(thresholds):
        df_tmp = df[df.isup_grade == label].reset_index(drop=True)
        gap_tmp = gap[df.isup_grade == label].reset_index(drop=True)
        
        df_remove_tmp = df_tmp[gap_tmp &gt; thresh].reset_index(drop=True)
        df_keep_tmp = df_tmp[gap_tmp &lt;= thresh].reset_index(drop=True)
        
        df_removes.append(df_remove_tmp)
        df_keeps.append(df_keep_tmp)
    
    df_keep = pd.concat(df_keeps, axis=0)
    df_removed = pd.concat(df_removes, axis=0)
    return df_keep, df_removed

def remove_noisy3(df, thresholds_rad, thresholds_ka):
    df_r = df[df.data_provider == "radboud"].reset_index(drop=True)
    df_k = df[df.data_provider != "radboud"].reset_index(drop=True)
    
    dfs = [df_r, df_k]
    thresholds = [thresholds_rad, thresholds_ka]
    df_keeps = list()
    df_removes = list()
    
    for df_tmp, thresholds_tmp in zip(dfs, thresholds):
        df_keep_tmp, df_remove_tmp = remove_noisy2(df_tmp, thresholds_tmp)
        df_keeps.append(df_keep_tmp)
        df_removes.append(df_remove_tmp)
    
    df_keep = pd.concat(df_keeps, axis=0)
    df_removed = pd.concat(df_removes, axis=0)
    return df_keep, df_removed

# Change thresh each label each dataprovider
thresholds_rad=[1.3, 0.8, 0.8, 0.8, 0.8, 1.3]
thresholds_ka=[1.5, 1.0, 1.0, 1.0, 1.0, 1.5]

df_keep, df_removed = remove_noisy3(df, thresholds_rad=thresholds_rad, thresholds_ka=thresholds_ka)
show_keep_remove(df, df_keep, df_removed)
```


