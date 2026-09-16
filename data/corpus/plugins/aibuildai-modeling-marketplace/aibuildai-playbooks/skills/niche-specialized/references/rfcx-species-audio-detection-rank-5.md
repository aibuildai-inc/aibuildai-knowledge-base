# 5th place solution (Training Strategy)

Competition: rfcx-species-audio-detection
Rank: #5
Source: https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/220432

# 5th place solution (Training Strategy)

Congratulations to all the participants, and thanks a lot to the organizers for this competition! This has been a very difficult but fun competition:)

In this thread, I introduce our approach about training strategy.
About ensemble part will be written by my team member.

Our team ensemble each best model.

My model is Resnet18 which has a SED header. This model's LWLRAP is Public LB=0.949 /Private LB=0.951, and I trained by google colab using [Theo Viel's npz dataset](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/198048)(32 kHz, 128 mels). Thank you, Theo Viel!!

Our approach has 3 stage, 
Other team members are different in some things likes the base model and hyperparameter, but these default strategies are about the same.

## 1st stage: pre-train

*※I think this part is not important. Team member Ahmet skips this part.*

This stage transfers learning from Imagenet to spectograms.

Theo Viel's npz dataset can be regarded as 128x3751 size image.
I cut to 512 by this image in sound point from t_min and t_max.
I train this image by tp_train and 30 sampled fp_train.

Parameters:
- Adam
- learning_rate=1e-3
- CosineAnnealingLR(max_T=10)
- epoch=50

Continue 2nd and 3rd stage use this trained weight.

## 2nd stage: pseudo label re-labeling

The purpose of stage2 is to improve the model and make pseudo labels by this model.

Use 1st stage trained weight.

The key point I think is to calculate gradient loss only labeled frame. The positive labels were sampled from tp_train.csv only and the negative labels were sampled from fp_train.csv only.
I put 1 to positive label and -1 to negative label.

```python
tp_dict = {}
for recording_id, df in train_tp.groupby("recording_id"):
    tp_dict[recording_id+"_posi"] = df.values[:, [1,3,4,5,6]]

fp_dict = {}
for recording_id, df in train_fp.groupby("recording_id"):
    fp_dict[recording_id+"_nega"] = df.values[:, [1,3,4,5,6]]
    
def extract_seq_label(label, value):
    seq_label = np.zeros((24, 3751))  # label, sequence
    middle = np.ones(24) * -1
    for species_id, t_min, f_min, t_max, f_max in label:
        h, t = int(3751*(t_min/60)), int(3751*(t_max/60))
        m = (t + h)//2
        middle[species_id] = m
        seq_label[species_id, h:t] = value
    return seq_label, middle.astype(int)

# extract positive label and middle point
fname = "00204008d" + "_posi"
posi_label, posi_middle = extract_seq_label(tp_dict[fname], 1) 

# extract negative label and middle point
fname = "00204008d" + "_nega"
nega_label, nega_middle = extract_seq_label(fp_dict[fname], -1) 
```

loss function is that:

```python
def rfcx_2nd_criterion(outputs, targets):
    clipwise_preds_att_ti = outputs["clipwise_preds_att_ti"]
    posi_label = ((targets == 1).sum(2) > 0).float().to(device)
    nega_label = ((targets == -1).sum(2) > 0).float().to(device)
    posi_y = torch.ones(clipwise_preds_att_ti.shape).to(device)
    nega_y = torch.zeros(clipwise_preds_att_ti.shape).to(device)
    posi_loss = nn.BCEWithLogitsLoss(reduction="none")(clipwise_preds_att_ti, posi_y)
    nega_loss = nn.BCEWithLogitsLoss(reduction="none")(clipwise_preds_att_ti, nega_y)
    posi_loss = (posi_loss * posi_label).sum()
    nega_loss = (nega_loss * nega_label).sum()
    loss = posi_loss + nega_loss
    return loss
```

And image are cut and stack by sliding window.
I set the window size to 512 and cut out the entire range of 60's audio data by covering it little by little. Cover 49 pixels each, considering that important sounds may be located at the boundaries of the division.

```python
N_SPLIT_IMG = 8
WINDOW = 512
COVER = 49

slide_img_pos = [[0, WINDOW]]
for idx in range(1, N_SPLIT_IMG):
    h, t = slide_img_pos[idx-1][0], slide_img_pos[idx-1][1]
    h = t - COVER
    t = h + WINDOW
    slide_img_pos.append([h, t])

print(slide_img_pos)
# [[0, 512], [463, 975], [926, 1438], [1389, 1901], [1852, 2364], [2315, 2827], [2778, 3290], [3241, 3753]]
```

I predict each sliding window and put the pseudo label, so I got 8 windows in one 60 sec recording.

|patch idx|pixcel|time(s)|
|--|--|--|
|0|0〜512|0〜8|
|1|463〜975|7〜15|
|2|926〜1438|14〜23|
|3|1389〜1901|22〜30|
|4|1852〜2364|29〜37|
|5|2315〜2827|37〜45|
|6|2778〜3290|44〜52|
|7|3241〜3753|51〜60|


Parameters:
- Adam
- learning_rate=3e-4
- CosineAnnealingLR(max_T=5)
- epoch=5

## 3rd stage: train by label re-labeled
This stage trains on the new labels re-labeled by 2nd stage model.

Use 1st stage trained weight.

The new label is ensemble by our team output like my 2nd stage.
- our prediction average value is
  - `>0.5`: soft positive = 2
  - `<0.01`: soft negative = -2

In this stage, I calculate gradient loss only labeled frame as with 2nd stage. 

Parameters:
- Adam
- learning_rate=3e-4
- CosineAnnealingLR(max_T=5)
- epoch=5

Some My Tips:
- Don't use soft negative.
- The re-label's loss(soft positive) is weighted 0.5.
- last layer mixup(from [this blog](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))

## CV

I use [iterative-stratification](https://github.com/trent-b/iterative-stratification)'s MultilabelStratifiedKFold. Validation data is made from tp_train only and fp_train data is used training in all fold.

Each stage LWLRAP is that:

|stage|CV|Public|Private|
|--|--|--|--|
|1st|0.7889|0.842|0.865|
|2nd|0.7766|0.874|0.878|
|3rd|0.7887|0.949|0.951|

3rd stage's re-labeled LWRAP is 0.9621.


## predict

In test time, I increase COVER to 256, so I got 14 windows in one 60 sec recording.
The prediction is max pooling in each patch.

I use clipwise_output in training, and I use framewise_output in prediction. This approach came from [shinmura0's discussion thread](https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/209684). Thank you  shinmura0:)

## did not work for my model
- TTA
- 26 classes (divide song_type)
- label wright loss
- label smoothing(but team member's kuto improved)

---

Finally, I would like to thank the team members.
If I was alone, I couldn't get these result.
kuto, Ahmet, thank you very much.

My code:
https://github.com/trtd56/RFCX
