# 3rd place Solution - Matcha & Object Detection

Competition: benetech-making-graphs-accessible
Rank: #3
Source: https://www.kaggle.com/c/benetech-making-graphs-accessible/discussion/418420

First of all, congrats to all the participants especially to those that managed to stick it to the end. This was overall a really interesting challenge, and we really enjoyed tackling it. Anyone in 0.86+ public scores could've taken the win so we're glad we dropped only 2 spots.

Thanks to @crodoc for the great teaming experience, we had two different opinions on how to tackle the problem : he went for and end-to-end approach whereas I wanted to do object detection & OCR. This proved to be ideal for this competition as object detection compensated for end-to-end weaknesses.

## Overview

<a href="https://ibb.co/qFXF9vq">[cls]</a>

Our solution is a two-step pipeline, where the first step is a simple classification task and the second step solves the task for the different chart types. For scatter and dot plots, we used a detection approach. For line and bar charts, Matcha was really strong. Here are our LB scores :

| |  Overall | Scatter | Dots | Line | Horizontal Bar | Vertical Bar|
| --- |
| **Public** | 0.87 | 0.09 | 0| 0.33 | 0.04| 0.39|
| **Private** | 0.71 | 0.28 | 0.01| 0.13 | 0.01 | 0.27 |

## Validation & Insights

From the description page : 
> The extracted figures in the training and public test sets are drawn from the same set of sources. The figures in the private test set are drawn from a distinct set of sources.

Public LB is leaky, we need to create models that generalize to new data sources. Therefore, we use all the extracted data as validation, and train only on generated data. Data provided by the hosts is not diverse enough to allow for generalization so we generated our own. Matcha and classification models were fullfit on all the data after careful parameter optimization, but detection models were not trained on any extracted data.

For validating on dots, we used a small curated dataset of 20 images found on google images. We estimated via probing that there are between 100 and 125 dots in the test set, all of them most likely being in the private test set. We did not probe for scatter though, this would've been helpful. 

## Step 1 - Classification

Nothing too fancy here, we trained models on (benetech + theo +  crodoc) generated data.

**Main parameters:**
- 2 epochs on 88k images. 
- lr `3e-4` or `5e-4`  (2 lrs for blend, we also used 2 seeds).
- Mixup and some color augmentations mostly.
- 256x384 image size.
- NfNet-l2 with 0.2 dropout.

## Step 2.a - Scatter

<a href="https://ibb.co/5v4cM9C">[scatter-pipe]</a>

For scatter, we relied on [YoloX](https://github.com/Megvii-BaseDetection/YOLOX) to detect all markers. Cached is used to take care of the other useful elements. If all the points are detected correctly, it’s not that complicated to infer the target : detect the ticks & labels, read the labels, and interpolate !

**More details:**
- Ensemble YoloX-m and YoloX-l with NMS, helps reduce the number of FNs.
- Models are trained on scatter + dot generated benetech data + a bunch of plots I generated and pseudo labeled, for 10 epochs.
- CV 0.67, public LB 0.09~, private 0.29 - performance drop almost exclusively comes from overlapping/too hard to detect markers.
- Bunch of post-processing to make the pipeline more robust to OCR mistakes and detection mistakes.
- We were initially using Yolo-v7, but had to switch to YoloX because of the first rule change. It took us a week to match Yolo-v7 performance with YoloX.

## Step 2.b - Dots

<a href="https://ibb.co/sgMLzxG">[dot-pipe]</a>

Dot pipeline is similar to scatter, but a bit more simple. We detect the points and cluster them, and map them to the detected x-labels. Labels with no assigned cluster are given the target 0, others the number of detected points. Counting points is not really robust to detection mistakes, so instead we used the height of the uppermost point and interpolated.

## Step 2.c - Bars & Lines

Matcha was really powerful here. We used `matcha-base` and set `is_vqa=False` to avoid giving texts as input to the model.

We trained Matcha to predict chart type, xs and ys for an image. The ground truth looked the same as @nbroad used for his donut approach (except we dropped the prompt token). We tried other approaches, but this worked best :
```
x_str = X_START + ";".join(list(map(str, xs))) + X_END
y_str = Y_START + ";".join(list(map(str, ys))) + Y_END
ground_truth = '<' + chart_type + '>' + x_str + y_str
```
e.g: `<line><x_start>0;2;4;6<x_end><y_start>2.7;2.2;3.6;5.2;<y_end>`

The most valuable boost we got by generating additional charts using matplotlib. We reused the values and texts from the train dataset to generate ticks & values, with different styles/patterns/fonts/colors for diversity. The code for generating additional images has around 1000 lines and basically covers most cases where the model was failing when validated on the "extracted" dataset (e.g. negative values, line edges, missing bars, multiline text, text rotations).

**Other things that helped :**
- Fix the number of decimal points for numerical values using their range :
`number_of_decimals = max(0, round(np.log10(1/ (max(y_ticks) - min(y_ticks))) + 3)`
- We use an additional chart type : histograms, to learn that such charts have one less y value. 
- Add an additional cross entropy loss for chart types.
- Ensemble several (4) models  (+0.01 public) :
   - Voting to compute the number of outputs and fix obvious mistakes.
   - Voting for categorical predictions.
   - Averaging for continuous predictions.

**More details:**
- lr 3e-5, cosine with 0.25 cycle.
- 10 Epochs.
- Save weights every 0.25 epochs and use a model soup of all checkpoints from epoch > 1.
- Augs : Color transforms, image compression and random scaling.

## Final Words

We were greatly perturbed by the unexpected second rule change, especially considering the fact that I was on vacation. Matcha and classification models were retrained on ICDAR during the last 3 days, this gave a 0.01 public LB boost but nothing on private. 

Thanks for reading !
