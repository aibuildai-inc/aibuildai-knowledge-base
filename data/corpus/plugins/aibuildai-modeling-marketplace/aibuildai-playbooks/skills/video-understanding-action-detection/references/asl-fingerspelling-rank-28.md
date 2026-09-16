# [28th Place Solution] Pre Training on Supplemental Data gives 0.01 LB Improvement

Competition: asl-fingerspelling
Rank: #28
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/434896

## Acknowledgements

Firstly, I'd like to extend my deepest gratitude to my teammates:
- @chellyfan
- @chuansj

Their enormous effort has been instrumental in this competition. 

I'd like also thanks to the authors of following notebooks. Our success wouldn't have been possible without the insights and techniques we adopted from these notebooks. A big thanks to you.

- **@irohith** for his excellent preprocessing and training notebooks:
  - [ASLFR CTC based on prev comp 1st place](https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place)
  - [ASLFR preprocess dataset TFRecords mean std](https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset-tfrecords-mean-std)

- **@hoyso48** for his 1st place solution from a previous competition:
  - [1st Place Solution Training](https://www.kaggle.com/code/hoyso48/1st-place-solution-training)

- **@greysnow** for the CTC on TPU notebook:
  - [ASLFR CTC on TPU](https://www.kaggle.com/code/shlomoron/aslfr-ctc-on-tpu)

- **@royalacecat** for the notebook on more blocks with quantization:
  - [The deeper, the better](https://www.kaggle.com/code/royalacecat/the-deeper-the-better)

## TL;DR

Our solution is based on the CTC open source notebook. Here are the key improvements we introduced (all scores are public LB):

- Only filter **full NaN frames** when generating tfrecord files on data preprocessing. (0.687->0.705)
- Adding more numbers of block in the model as described in [The deeper, the better](https://www.kaggle.com/code/royalacecat/the-deeper-the-better). (0.705->0.716)
- **Increased frame length** from 128 to 256. (0.716->0.746)
- **Enhanced feature selection** by adding more face and pose indices, increase from 96 to 128 indices. (0.746->0.763)
- Incorporated **CTC loss with label smoothing** for added regularization. (0.763 ->0.766)
- **Pretrained our model** on the supplemental dataset. Pre train 80 epochs on supply set then 80 epochs on main dataset. (0.766->0.777)
- Adopted the **motion feature** technique from a previous competition. (0.777->0.781)
- Increase training epochs on main dataset from 80 to 120 (0.781->0.782)

## Data Preprocessing

Referring to @irohith's preprocessing notebook, we made changes by adding more face and pose indices. We selected the new indices based on their variance — higher variance indicates more information.

Initially, the indices were:

```python
LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE
```
We then updated them to:

```python

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]

# New face_id indices
face_id = [454,356,323,361,389,288,251,264,447,366,368,
           401,397,435,284,301,372,345,383,367,365,352,433,
           376,298,265,93,234,300,132,340,353,127]

# Append new face_id indices to LIP if not already present
for k in face_id:
    if k not in LIP:
        LIP.append(k)

# Trim the LIP list
l = len(LIP)
LIP  = LIP[:int(l - l/4)]

# Updated POSE indices
LPOSE = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RPOSE = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
POSE = LPOSE + RPOSE
```
filter out full Nan frame

origin notebook:

```python
hand = tf.concat([rhand, lhand], axis=1)
hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)
```
update to 

```python
hand = tf.concat([rhand, lhand,lip,rpose,lpose], axis=1)
hand = tf.where(tf.math.is_nan(hand), 0.0, hand)
mask = tf.math.not_equal(tf.reduce_sum(hand, axis=[1, 2]), 0.0)
```
Those new indices along with 256 frame length gives us about 0.02 LB improvement!

## Motion feature
see  [1st Place Solution Training](https://www.kaggle.com/code/hoyso48/1st-place-solution-training)


## CTC Loss with Label Smoothing
To enhance our model's stability during training, we implemented a CTC loss with an added regularization component. By combining the CTC loss with a Kullback-Leibler (KL) divergence, we introduced label smoothing to the model. Smooth weight = 0.7 works best for our model
```python
def smooth_ctc_loss(labels, logits, num_classes = 60 , blank=0, weight=0.7):
    # Compute CTC Loss
    label_length = tf.reduce_sum(tf.cast(labels != pad_token_idx, tf.int32), axis=-1)
    logit_length = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
    ctc_loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_length,
        logit_length=logit_length,
        blank_index=pad_token_idx,
        logits_time_major=False
    )
    ctc_loss = tf.reduce_mean(ctc_loss)

    # Compute KL Divergence Loss
    kl_inp = tf.nn.softmax(logits)

    # Create the target distribution
    kl_tar = tf.fill(tf.shape(logits), 1. / num_classes)

    # Compute the KL divergence
    kldiv_loss = (tf.keras.losses.KLDivergence(tf.keras.losses.Reduction.NONE)(kl_tar, kl_inp) 
                 + tf.keras.losses.KLDivergence(tf.keras.losses.Reduction.NONE)(kl_inp, kl_tar))/2.0

    kldiv_loss = tf.reduce_mean(kldiv_loss)

    # Combined Loss
    loss = (1. - weight) * ctc_loss + weight * kldiv_loss
    return loss
```
## Pre-training the Model on the Supplemental Dataset

During our experimentation, we found a significant insight: training solely on the **supplemental dataset** yielded a LB score of 0.367. This highlighted the potential value embedded within the supplemental data.

We decided to first pre-train our model on the supplemental dataset, then loaded these pre-trained weights to further train the model on the main dataset. We tested multiple epochs: 40, 60, and 80 for pre-train. 80 epochs works best and give a improvement of 0.013 on the LB score.

## What We Tried, But Didn't Work

1. **Augmentation**:
   - **Random Affine Transformation**
   - **Random Part Removal**: We tried dropping certain parts (like lip, lpose, rpose, lhand, rhand) by setting them to NaN with a 5% probability. 
   - **Feature Switch for Phrases**: switch features for phrases within the same category.

   In 2nd place solution, @hoyso48 mention that some augmentation didn't improve the model during short epoch training but in longer training would have the effect. This could applied to our case as well but we haven't verified yet.

2. **Switching to GIC-CTC**:
   - **Reference Paper**: [Improving CTC-Based ASR Models With Gated Interlayer Collaboration](https://ieeexplore.ieee.org/document/10094820)
   - Increasing parameters led to our model exceeding the 5-hour inference time limit.


## What We could Imporve
1. long epochs training. We only train 120 epochs for final submission, which is much less compare to other top place solution (usually 300-500 epochs). We already saw a 0.001 LB improve from 80 epochs to 120 epochs on the final day, but it's too late
2. Adjust the leaning rate and weight decay more carefully. We simply use the cosine decay lr scheduler in the public notebook and set weight decay to 0.05
3. Utilized AWP for longer epoch training

## Notebook:
 
https://www.kaggle.com/code/wuhongrui/aslfr-gcs-path
https://www.kaggle.com/wuhongrui/aslfr-training
https://www.kaggle.com/wuhongrui/ctc-final-inference-notebook

Training notebook is written specific for training on Google Colab. Frist use gcs-path notebook check the path in training notebook. Set USE_SUPPLY = True first to train supply set, then set  USE_VAL = True to load pre-train weight and train main data set.

To use it on Kaggle TPU, change the file address to corresponding Kaggle input address and you NEED to modify the CTC loss. Check [ASLFR CTC on TPU](https://www.kaggle.com/code/shlomoron/aslfr-ctc-on-tpu)
