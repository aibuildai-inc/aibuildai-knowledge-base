# 3rd place solution & become GM!! (updated with code)

Competition: rsna-intracranial-hemorrhage-detection
Rank: #3
Source: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117223

## update
code is [here](https://github.com/okotaku/kaggle_rsna2019_3rd_solution).

Hi, dear kagglers. First of all, thank you very much RSNA and kaggle for hosting such a fantastic competition. And congrats winners and all kagglers:) 
I finally became kaggle Grandmaster. It was a super tough road but all experience made me stronger. I am very proud of it😆
 
Here is my solution. I will write details in later parts and will share my github repo after I clean up it.



Final model: private 0.045
User stacking model only: private 0.043 (I couldn't select it qq)

## Special Preprocessing
### windowing
I used 2 types of windowing.
- [subdural window](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/110728#latest-659011)
- [Appian’s 3 types windows](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage/blob/master/src/cnn/dataset/custom_dataset.py#L16)

For me subdural was a little better.

### concat user slice
This method gave me much improvement. There are some images (about20 - 40) in one SeriesInstanceUID. And when sorted by ImagePositionPatient2, you can see that targets are continuous. I will call those images, s1, s2, s3, ..., st, st+1, … in my post.
Here is the example. You can see more details in [my kernel](https://www.kaggle.com/takuok/eda-of-rsna).



So I decided to concat some images from the same SeriesInstanceUID. 
- st-1, st, st+1
- st, st+1, st+2
- st-2, st-1, st
- st-2, st, st+2
- np.mean (st-3, st-2, st-1), st, np.mean(st+1, st+2, st+3)
- np.mean (st-5, st-4, st-3, st-2, st-1), st, np.mean(st+1, st+2, st+3, st+4, st+5)
- np.mean (st-X for X in all values), st, np.mean(st+X for X in all values)

Then predicted st’s target.

And I tried multi task training.
- st-1, st, st+1 then predict targets of st-1, st, st+1
- st-2, st, st+2 then predict targets of st-2, st, st+2

This model got 0.060~0.062(sry I forgot) on stage1 Public. It was my best single model and those 2 models improved my ensemble score from 0.057 to 0.056 on stage1.

## User Stacking
I used “concat user slice” method to show models multiple slices of the user. And I used this method on ensemble parts. I call it User Stacking.



## Other things I used
- They didn’t have much improvement, but I write up.
- Appian’s 0.066 models
- predict 5 classed and fill “any” on max prediction.
- [CQ500 External Data](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/113339#latest-664918)
- crop black area
- retrain stage2 data
- [generalized mean pooling](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065#latest-636669)

## What didn’t work.
- EfficientNet
- Cbam Resnet

*Slide design: Japanese Autumn leaves (紅葉: koyo)
