# 3rd Place Solution

Competition: asl-signs
Rank: #3
Source: https://www.kaggle.com/c/asl-signs/discussion/406568

We used an **ensemble of six conv1d models and two versions of transformers** based on the [public notebook](https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training). The key points are **data preprocessing, hard augmentation and ensemble**.

I initially tried to develop my own solution based on the aforementioned public transformer but then I found that the architecture of a model didn’t matter as much as proper work with data and ensembling, so I switched to more simple architectures like **conv1d multilayer models**. I used `Keras` and utilized all the advantages of native tflite layers like `DepthwiseConv1D`. My typical model looked like that:
```python
do = 0.5

model = Sequential()
model.add(InputLayer(input_shape=(max_len, 61, 2)))
model.add(Reshape((max_len, 61*2)))

model.add(Conv1D(64, 1, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu'))
model.add(BatchNormalization())

model.add(Conv1D(64, 1, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(DepthwiseConv1D(5, strides=2, padding='valid', depth_multiplier=4, activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool1D(2, 2))

model.add(Conv1D(256, 1, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(DepthwiseConv1D(3, strides=1, padding='valid', depth_multiplier=1, activation='relu'))
model.add(BatchNormalization())

model.add(Conv1D(256, 1, strides=1, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(DepthwiseConv1D(5, strides=2, padding='valid', depth_multiplier=4, activation='relu'))
model.add(BatchNormalization())

model.add(GlobalAvgPool1D())
model.add(Dropout(rate=do))

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=do))

model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=do))

model.add(Dense(250, activation='softmax'))
```

Then two of my colleagues joined me. We had a lot of ideas such as using synthetic data. We found this [paper](https://upcommons.upc.edu/bitstream/handle/2117/94313/TEPP1de1.pdf), created a script which generated synthetic 3D points of a hand parametrized by inner parameters (joints rotations) and camera model and tried to predict the inner parameters of hand by a model, based on the 3d points. And then use this pretrained model to preprocess data. But unfortunately we didn’t have enough time to finish it.

Also we tried to train on additional data (WLASL), found that this dataset gave ~0.005-0.007 increase in local CV score but then we abandoned this idea and haven’t used additional data in our LB submissions.

## Normalization
We used normalization by this reference points `ref = coords[:, [500, 501, 512, 513, 159,  386, 13,]]`
We didn’t use depth dimension.

We didn’t throw away frames without hands but we took the presence of a hand into account during TTA. 

## Augmentation
- Random rotation, shift, scale applied globally to each frame
- Random small shift added to every point 
- Combinations of parts - for example we took a hand from one sample and lips from another sample with the same label. This augmentation gave a significant increase in score
- CutMix - took parts of samples from different classes. For hand we used 0.7 as a label value, for other parts - 0.3

## TTA
- Random left/right padding for short sequences
- For large sequences of frames we throw away frames with different probability taking into account presence of hands on a frame

For different models we used different combinations of points: both hands / only active hand, lips, eyes, top part of pose. For some models we used only every second point of lips/eyes.
For models where we used only one hand we determined which hand is more presented in the video and if it was the left hand we mirrored all points and their coordinates.
We took only **32 frames** for all models except for one where we used 96 frames.

These conv1d models were fast and lightweight so we could ensemble up to six models and still had space and time for additional models and TTA. Ensemble of six such models got **0.7948** on public LB and **0.8711** on private LB.
Then we decided to collaborate with @sqqqqy who had been working on improving of transformer from a public kernel. His solution looked like that:

## 1. Transformer Model
- Our transformer model is based on the [public notebook](https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training), using different sequence normalization and smaller UNITS size to reduce model parameters.
- We implement two kinds of transformer-based model, the first model learn sperate embeddings for each part (just like public notebook), the second learn one embedding with input whole xyz sequence.
- 1seed of first model score **LB 0.77+**, 1seed of second model score **LB 0.768**

## 2. Preprocessing
- We use 20 lip points, 32 eyes points, 42 hands points(left hand and right hand) and 8 pose points.
- The input sequence is normalized with shoulder, hip, lip and eyes points.
- Filling the NaN values with 0.0
- Learn a motion embedding by input $(d_x, d_y, \sqrt{(d_x)^2+(d_y)^2})_t$ sequence, 
$$
(d_x, d_y)_t = xyz_t - xyz_{t-1}
$$
- The final embedding is the concat of motion embedding and xyz embedding

## 3.Augmentation
- Global augmentation (apply same aug for all frames), including rotation(-10,10), shift(-0.1,0.1), scale(0.8,1.2), shear(-1.0,1.0), flip(apply for some signs)

- Time-based augmentation (apply aug for some frames), random select some frames(1-8) do affine augmentations, random drop frames (fill with 0.0)

- The augmentations greatly improve the score

## 4. HyperParameters
- NUM_BLOCKS, 2
- NUM_HEAD 8
- lr 1e-3 
- Optimizer AdamW
- Epoch 100
- LateDropout 0.2-0.3
- Label Smoothing 0.5


Six conv1d models with TTA combined with two transformers gave us third place in this competition.
