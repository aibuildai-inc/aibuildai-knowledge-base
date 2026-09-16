# 43rd place solution

Competition: rfcx-species-audio-detection
Rank: #43
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220335

First of all, many thanks to Kaggler.
 I got a lot of ideas from Kaggler in the discussion. And it was fun.

My solution summary is below.
+ The high resolution of spectrogram
+ Post-processing with moving average
+ Teacher-student model for missing labels

#1st stage: SED
I started from basic experiment with [SED](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/211007). Why SED? Because I think SED is strong in the multi label task. 

I used log mel-spectrogram as the input of SED. Basic experiment involves data augmentation (Gaussian noise, SpecAugment and MixUP), backbone model choice and adjusting resolution of log mel-spectrogram. As a result, below condition was the best for me.
+ No noise injection
+ MixUp
+ The best model architecture is EfficientNet
+ **The higher the resolution** of log mel-spectrogram, the better the result.

### The resolution
The most important one is the resolution. Recently, in Kaggle computer vision solution, the higher the resolution of the image, the better the result. In spectrogram, the same phenomenon may happen. In mel-spectrogram, the resolution can be changed by adjusting "hop_size" and "mel_bins". Following result is changing the resolution with ResNest50(single model).

|Resolution(Width-Height)|public LB|
|---|---|
|1501-64(PANNs default)|0.692
|3001-128|0.805
|6001-64|0.725
|1501-256|0.761
|751-512|**0.823**
|1501-512|0.821

The resolution was critical! According to experimental result, good resolution was "high" and similar to the square. 751-512 looks good. As a result, I chose 858-850. This configuration is as follows.

```
model_config = {
    "sample_rate": 48000,
    "window_size": 1024,
    "hop_size": 560,
    "mel_bins": 850,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 24
}
```

### Post-processing
I used [Framewise output](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/211007) for submission. It contains time and classes information. But there is a lot of false positive information in framewise output. Because they are not processing by a long time information. Therefore **a short event of framewise output should be deleted.** I prepared post-processing for framewise output. It is a moving average.

[image.png]

By taking a moving average in the time direction for each class, we can delete short events. This idea is based on [the paper](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Chan_6.pdf)[1]. The sample code is as follows.
```
def post_processing(data): # data.shape = (24, 600) # (classes, time)
    result = []
    for i in range(len(data)):
        result.append(cv2.blur(data[i],(1,31)))
    return result
```

I improved LB by using moving average. The following result is comparing post-processing with EfficientNetB3(single model).

||public LB|
|---|---|
|w/o post-processing|0.785|
|w/ post-processing|**0.840**|
 
###Summary
+ MixUp(alpha=0.1)
+ Epoch 30
+ Adam(lr=0.001) + CosineAnnealing(T=10)
+ Batchsize 6
+ Use only tp label
+ Get random tp clip 10 sec
+ The resolution of log mel-spectrogram: 858-850
+ Loss function: BCE
+ [Weak label training](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/211007)
+ Post-processing: moving average

Then I got **0.916 public LB** with EfficientNetB0(5-folds average ensemble).

#2nd stage: missing labels and the ensemble
I reported [discovering missing labels and re-train](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/208830#1139171). And It didn't work. After that, I thought about missing labels again. My answer is that the model is not correct for discovering missing labels. There are a lot of missing labels around tp. Therefore the model is not correct. 

[train.png]

To solve this issue, I used teacher-student model. 

[geretation.png]

1st generation is similar to 1st stage. I gradually increased model prediction ratio. By using teacher-student model, I could discover missing labels. Specially, in strong label training, teacher-student model was effective. Following result is teacher-student model score with EfficientNetB0.

[image.png]

"MixUp rate" is probabilistic MixUp. This method is based on [the paper](https://arxiv.org/abs/2102.01243)[2]. 

Finally, I made the ensemble of 1st stage model and 2nd stage model. Ensemble procedure is simple average. Then I got **0.924 public LB.**

#References
[1] Teck Kai Chan, Cheng Siong Chin1 and Ye Li, "SEMI-SUPERVISED NMF-CNN FOR SOUND EVENT DETECTION".
[2] Yuan Gong, Yu-An Chung, and James Glass, "PSLA: Improving Audio Event Classification with
Pretraining, Sampling, Labeling, and Aggregation".

#Appendix: the resolution and EfficientNet
Finally, I show interesting result. It is relationship between EfficientNet and the resolution. The following result is public LB(5-folds average ensemble).

|Resolution(W-H)|751-512|751-751|858-850|
|---|---|---|---|
|EfficientNetB0|0.893|0.904|**0.916**|
|EfficientNetB3|**0.913**|0.912|0.900|

In B0, the higher resolution, the better result. But B3 was vice versa. Usually, the larger EfficientNet, the better at [high resolution](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154683) it is. But the above is reverse. Why?

Maybe **domain shift**(train: noisy sound -> test: clean sound) is concerned. B3 has learned about **train domain features**(noisy sound). On the other hand, B0 has less representational ability than B3. Therefore B0 has learned the **common features** of the train and test domain with high resolution. Without domain shift, B3 would have also shown good results with high resolution.
