# 4th place solution

Competition: rsna-str-pulmonary-embolism-detection
Rank: #4
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193970

First of all, Thank you to the organizers and Congrats to all the participants and winners !

My solution is very simple.
・train CNN backbone (Stage-1)
・extract embedding
・train LSTMs (Stage-2)



**Stage-1:**
　Split 3fold and train CNN which predict pre_present_on_image label for each images.
As a result of exploring several backbones, I decided to use rexnet200. Efficientnet-b4 and b5 got nan loss with mixed precision so I gave up using.
Full-size(512x512) jpeg images preprocessed by Ian Pan’s windowing are fed into CNN.
For augmentation, I used the following:
 ・Horizontal Flip
 ・ShiftScaleRotate
 ・One of(Cutout, GridDopout)
Since the ratio of pe present on image was quite small, I also use focal loss.
In Stage-1, I created 6 models. (3-fold BCE and 3-fold Focal) 

**Stage-2:**
　The embedding was dumped and sorted in z-axis order and stored in a disk, and then LSTMs were trained using it. I implemented a mini-batch training of variable length series since the number of images for each study is different. Each series is filled with invalid values until the maximum series length (1083) of the training data is reached, and the invalid part is ignored when calculating the loss.
　For each study, embeddings were put into BiLSTMx2 and each LSTM’s outputs went into two branches: one to predict pe_present_on_image and the other to predict the exam level label. In pe_present_on_image branch, the two outputs were simply added and transformed by Linear Layer. In exam_level branch,  features were aggregated using attention layer and  transformed by Linear Layer.
For Loss, I used simple BCE.

**Inference:**
I implemented a model that connects backbone and LSTMs for inference without dumping embedding to disk. Inference is done by batch_size=1. There were not enough time to apply TTA.

**Submit model:**
・6-model average ensemble. 
　3-fold with backbone trained by BCE Loss + 3-fold with backbone trained by Focal Loss.
　public LB: 0.159 private LB: 0.152
・3-model average ensemble.
　3-fold with backbone trained by BCE Loss
　public LB: 0.158 private LB: 0.153

**Thank you !**
This is my first gold medal. I’m super glad to finally be a kaggle master!
And this also is  my first time of posting discussion. 
I'm sorry if my English is not good enough to convey my solution.

The code link will appeare here once I clean it up.

Update:
[repository](https://github.com/piwafp0720/RSNA-STR-Pulmonary-Embolism-Detection)
[inference kernel](https://www.kaggle.com/kazumax0720/rsna-str-pulmonary-embolism-detection-inference)
