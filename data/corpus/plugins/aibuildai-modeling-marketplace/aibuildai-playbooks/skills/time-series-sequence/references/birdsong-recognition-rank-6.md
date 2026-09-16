# 6th place solution and some thoughts

Competition: birdsong-recognition
Rank: #6
Source: https://www.kaggle.com/c/birdsong-recognition/discussion/183204

First of all, I would like to sincerely thank [@stefankahl](https://www.kaggle.com/stefankahl), [@tomdenton](https://www.kaggle.com/tomdenton), [@holgerklinck](https://www.kaggle.com/holgerklinck), and the members of kaggle team for hosting this competition. I had a lot of fun tackling on some of the challenging problems of machine learning thinking of the generative process of the data. Also many thanks to [@hengck23](https://www.kaggle.com/hengck23) for actively sharing a lot of deep insights. It helped me a lot to come up with some good ideas and also made me convinced that I was in good direction.

Following the recent two competitions: [PANDA](https://www.kaggle.com/c/prostate-cancer-grade-assessment) challenge and [GWD](https://www.kaggle.com/c/global-wheat-detection) challenge, this competition was also about **domain shift** and **noisy labels**.
Combination of these two challenging topics made this competition extremely difficult and we were troubled a lot how to make stable validation scheme. To be honest, contrary to [@cpmpml](https://www.kaggle.com/cpmpml)'s [expectation](https://www.kaggle.com/c/birdsong-recognition/discussion/181499#1010570), I couldn't find any good local validation scheme as the labels of training dataset contains a lot of noise. `Trust LB` was also not a very good policy since public LB was only 27% and we didn't know how the test set was devided. Instead, I took the policy of [PANDA's competitors](https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169230) - **Ignore CV, care about public LB, and trust methodology**.

Although this competition was difficult as a data science competition, it was actually pretty close to a realistic setting and was full of problems that we often face in applying data science to real-world problem. Especially the combination of domain shift and noisy labels often happens (I think) when we are to use data from User Generated Contents(UGC) web service like Xeno Canto, YouTube, Twitter for training machine learning algorithms. Therefore, I think my solution is useful not only for this competition but also for those data science tasks related with UGC data, as it's basically focused on dealing with noisy labels and domain shift.

## Solution in three lines

* 3 stages of training to gradually remove noise in labels
* SED style training and inference as I introduced [here](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection)
* Ensemble of 11 EMA models trained with the whole dataset / whole [extended dataset](https://www.kaggle.com/c/birdsong-recognition/discussion/159970) to stabilize the result

Code is available here: [koukyo1994/kaggle-birdcall-6th-place](https://github.com/koukyo1994/kaggle-birdcall-6th-place).
Note: I've re-implemented the code from original repository since it's quite messy. However, I haven't checked whether whole pipeline works; if you find something wrong with the code, please let me know :)

## Motivation

It's quite obvious that there is a huge gap between training dataset and test dataset: it was described by the host and can also be seen through submission. However, *how* they are different is not obvious. Therefore, the first thing I did was to understand by observation what kind of domain shifts are seen between training and test datasets.

Domain shift is an umbrella term and there are several problem classes of domain shift. The most famous one is *covariate shift*, where \\( P_{train}(X)\neq P_{test}(X) \\) and \\( P_{train}(Y|X)=P_{test}(Y|X) \\). This one is well-studied and several algorithms are proposed to deal with the situation, but is not the type of domain shift in this competition. Another problem class is *prior probability shift* or *target shift*, where \\( P_{train}(Y)\neq P_{test}(Y) \\) and \\( P_{train}(X|Y)=P_{test}(X|Y) \\), also not the type in this competition because \\( P_{train}(X|Y) \\) is not the same as \\( P_{test}(X|Y) \\). In fact, in this competition, multiple distribution shifts are present - shift in input space, shift in prior probability of labels, and shift in the function which connects \\( X \\) and \\( Y \\).

How should we tackle a problem with various distribution shifts? The answer is simple - *divide the difficulty*. As I wrote above, shifts were three folds:

1) shift in input space. For example, SNR difference or difference in sound collection environment (device/sampling rate/temperature/weather/...) between train and test. Occurence of non-target sound events is also a part of this shift.
2) shift in prior probability of labels. Distribution difference of species or distribution difference of calltypes, or else.
3) shift in the function which connects \\( X \\) and \\( Y \\). This has a very strong relation with label noise. Thinking of how the labels were created in train dataset and in test dataset, one could come up with the fact that Label Function(LF) of train dataset is completely different from that of test dataset. The former is annotations of the uploader (and can have large variation), whereas the latter is probably those of dedicated annotator(s) (and possibly have smaller variation).

I decided to address these one by one and applied the following techniques.

* For 1), providing all the possible variation for train dataset may help. This is done by data augmentation.
* For 2), I just couldn't come up with smart ideas. I used ensemble of multiple models trained with datasets with different distributions of the labels to address this, but I think that is suboptimal.
* For 3), correcting the labels of train dataset to make train LF closer to test LF can help.

On the other hand, label noise is also a term that contains multiple problem classes. First, in this competition, labels of train dataset are provided as *weak labels*. As @hengck23 pointed out [here](https://www.kaggle.com/c/birdsong-recognition/discussion/174774#972122), weak label can be treated as noisy label if we change the point of view. Also there are some missing labels, which I'll explain later.

I also decided to address these one by one.

* For weak label as noisy label, I first train a model with long chunk and then use the prediction as corrected label. I at first tried to create strong labels but couldn't make the first stage model good enough, therefore I used the prediction to correct weak labels we have in train dataset.
* For missing label, I also used the prediction of a model to find those.

## First stage - build a model useful enough for addressing missing labels

In this stage, I used PANNs model. I used some basic augmentations (`NoiseInjection`, `PitchShift`, `RandomVolume`) and used `secondary_labels`. The keys in this stage were two folds:

* train with long chunk(30s) so that it would include call events of the species in `primaly_label` and `secondary_labels`
* use attention pooling and max pooling to get weak prediction from `framewise_output`

Here are the reason behind.

### train with long chunk

Assume we have `primary_label` of `birdA` and `secondary_labels` of `birdB` and `birdC`. Melspectrogram of the corresponding audio clip is something like the figure below (sorry for my poor drawing). If we use small window size, it may not include any sound events or include some sound events but not enough for the given labels. To make the model learn correctly, we need to make each label correspond to call event(s) of each species. For this reason, I used long chunk. Maybe it is better to use longer chunk like 1 minutes or more but I compromised to use 30s chunk considering the time for computation.



### Combination of attention pooling and max pooling to get weak prediction

This is something I shared [here](https://www.kaggle.com/c/birdsong-recognition/discussion/167611).
In the comments in [my SED notebook](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection), some said there were lots of false positives. This comes from the following reason.

Weak prediction from attention pooling is made by applying self-attention filter on `framewise_outputs`. Now, we only have weak labels, we calculate the loss with weak predictions and weak labels. The gradient will be distributed to self-attention layer and pointwise classifier, but for pointwise classifier, strong supervision comes when self-attention put high probability at that point. Therefore, pointwise classifier are more likely to produce high probability value and rely on the attention layer to reduce false positives. This is not good when we are interested in the output of pointwise classifier(`framewise_outputs`).

On the contrary, max pooling suppresses high probability values come out from pointwise classifier but it also has a defect that it is weak to impulse noise. Therefore, combination of max pooling and attention pooling can make balanced prediction and we can expect that be good prediction. For this reason, I used both but not combining the weak prediction from each aggregation but use each output and calculate loss for each, and sum them up. Therefore, the loss function I used in this stage was like this

```python
bce = BCELoss()
loss_att = bce(weak_pred_with_attention, label)
loss_max = bce(weak_pred_with_maxpooling, label)
loss = 1.0 * loss_att + 0.5 * loss_max
```

### Summarize this stage

* Single PANNs model
* BCE on `clipwise_output` and also on maxpooled output.
* Adam + CosineAnnealing, 55epochs training
* train with randomly cropped 30s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform
  - `NoiseInjection` (max noise amplitude 0.04)
  - `PitchShift` (max pitch level 3)
  - `RandomVolume` (max db level 4)

### Get oof prediction and use it to find missing labels

With training procedure above, the model would get around 0.575 - 0.578. In fact, this is the weight I used in the public notebook.
I trained 5folds and got oof prediction on the whole training set. Then I used this oof prediction to find missing labels.

Missing labels are more likely to be found from samples that does not have `secondary_labels`. It is up to the uploader to fill in `secondary_labels` or `background`, so some uploaders may not feel like to fill in those. Therefore, I picked samples without `secondary_labels` and used oof prediction of those to get additional labels if the probability of species that are not in their `primary_label` is over 0.9.

## Second stage - build a model with additional labels to get stronger labels

In this stage, I used SED model with ResNeSt encoder. The difference between first stage and second stage is not that big - only the model, the existence of found labels (the labels obtained from the oof prediction of the first stage), and the input. I started to use 3channel input. The first channel was normal log-melspectrogram and the second channel was PCEN. The third channel was also log-melspectrogram but instead of using `librosa.power_to_db(melspec)`, I used `librosa.power_to_db(melspec ** 1.5)`. The idea of using different input for each channel comes from [this post](https://www.kaggle.com/c/birdsong-recognition/discussion/170959). The chunk size is also reduced to 20s because of the GPU memory size limitation.

I also changed the attention pooling slightly given the [advice of @hengck23](https://www.kaggle.com/hidehisaarai1213/introduction-to-sound-event-detection/comments) to use `torch.tanh` instead of `torch.clamp`.

The result of this stage got around 0.60. I also trained 5folds model and this time I got oof prediction of `framewise_outputs`.

To summarize,

* SED model with ResNeSt50 encoder, attention pooling head of PANNs (`torch.clamp` -> `torch.tanh`)
* BCE on `clipwise_output` and also on maxpooled output
* Adam + CosineAnnealing, 75epochs training
* Add additional `secondary_labels` found in stage1
* 3channels input - \[normal logmel, PCEN, `librosa.power_to_db(melspec ** 1.5)`\]
* train with randomly cropped 20s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform
  - `NoiseInjection` (max noise amplitude 0.04)
  - `PitchShift` (max pitch level 3)
  - `RandomVolume` (max db level 4)

## Third stage - build a model with oof framewise_outputs

Despite the use of training with long chunk size and missing labels, label noise problem is far from solved. SNR level to decide whether a call event was present is different between training set and test set. In fact, that was also different between samples of training set because basically each annotator (uploader) had their own labeling criteria. For this reason, I used the oof prediction of `framewise_outputs` of the second stage to further correct the labels of the training dataset. 

In this stage, I also cropped 20s chunk randomly to get a batch. At this time I also cropped the corresponding part of the predicted `framewise_outputs` and apply threshold on them (threshold varies from 0.3 - 0.7). After thresholding, I get max pooling of the thresholded prediction in time axis to get chunk level prediction. Still, this prediction can be noisy and may contain false positives, I apply `logical_and` between predicted labels and provided labels. In this way, I got corrected chunk level label and use that for training.

Here's the pseudo-code of the process above

```python
y_batch = y[start_index:end_index]
soft_label_batch = soft_label[start_index_for_label:end_index_for_label]  # (n_frames, 264)
thresholded = (soft_label_batch >= threshold).astype(int)

weak_pred = thresholded.max(axis=0)  # (264,)
corrected_label = np.logical_and(label, weak_pred)  # (264,)
```

In this stage, I also tried EfficientNet-B0 encoder and FocalLoss. Combination of ResNeSt encoder and FocalLoss didn't work well, whereas EffNet-B0 and FocalLoss worked well on public LB.

All the other settings were the same as that of second stage. After this, I got around 0.61x score on public LB.

To summarize,

* SED model with ResNeSt50 encoder, attention pooling head of PANNs (`torch.clamp` -> `torch.tanh`) / EfficientNet-B0 encoder
* BCE on `clipwise_output` and also on maxpooled output / FocalLoss on `clipwise_output` and also on maxpooled output for EfficientNet-B0
* Adam + CosineAnnealing, 75epochs training
* Add additional `secondary_labels` found in stage1
* Correct labels using the prediction of stage2 model.
* 3channels input - \[normal logmel, PCEN, `librosa.power_to_db(melspec ** 1.5)`\]
* train with randomly cropped 20s chunk
* validate on randomly cropped 30s chunk
* Augmentations on raw waveform
  - `NoiseInjection` (max noise amplitude 0.04)
  - `PitchShift` (max pitch level 3)
  - `RandomVolume` (max db level 4)

## Ensemble

With the corrected chunk level label, I trained the model with the whole dataset and use EMA model (using the implementation [here](https://pytorch.org/docs/stable/optim.html#stochastic-weight-averaging)) for inference. This is a technique also used in [GWD competition](https://www.kaggle.com/c/global-wheat-detection/discussion/172458). 

I prepared models with different threshold (threshold on `framewise_outputs` of oof prediction) to make the model robust. Also important was the use of extended dataset. It doesn't get better result on public LB but when I used that for ensemble, it bumped up the score.

## Things that didn't work for me

* Mixup
* Calltype classification (781 class)
* noisy student training
* Larger models of efficientnet (b1, b2, b3...)
* etc...

## Things that worked but doesn't make sense to me

When I change the PERIOD used for inference, the result greatly changed. Basically the larger value PERIOD is, the better the result. I just couldn't figure out why.

## Things that I wanted to try but couldn't

There are lots of things I couldn't do due to the time limitation 

* Use a variety of models
* Further refinement of the labels of train dataset/train extended dataset
* Use of auxiliary predictor to predict longitude/latitude/elevation and use the prediction to correct the main classifier
* Post-processing to refine prediction using species correlation information (I couldn't get the API key for ebird.org therefore I couldn't collect correlation information)
* Mixing background noise
* etc...

## The thing that helped me a lot during competition

I created a simple streamlit app to check the audio data. I mainly used this to check the effect of augmentations or to check the quality of SED models prediction. This helped me a lot to figure out the major problems of this competition

https://github.com/koukyo1994/streamlit-audio

Later I learned @fkubota also made an app with similar functionality. I didn't use this but it seems it's better than mine.

https://github.com/fkubota/spectrogram-tree
