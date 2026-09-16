# 9th place - Custom CUDA kernel for wave propagation

Competition: waveform-inversion
Rank: #9
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587498

Thanks to kaggle and everyone involved for hosting this exciting competition. It was a great learning experience.

## Introduction
I joined the competition only 2 weeks before end and wanted to make the most of the time in terms of learning experience. Coming from two computer vision competitions before, I was not very interesting in spending a lot of time on the modeling side. So I basically took over the two main models and hyperparameters from @brendanartley and focused on other aspects of the competition. My main focus was to do efficient experimentation under time pressure when training takes a long time. I also wanted to try an "idle speculation" to just add some spice to this competition. 

## Cross validation
For cross-validation I simple split the data by file into 8 folds. In the first week I only trained and evaluated on CurveFault_B since this seemed to be the hardest, yet representative class.

## Model

As said in the introduction, my models are copied from @brendanartley. I used the `caformer_b36.sail_in22k_ft_in1k` and same convnext set-up, although adjusted to run a bigger version, `convnextv2_huge.fcmae` . 

## Training

### General setup
The main difference to other competitors is probably the training part. On the one hand I wanted to recyle training runs as much as possible as MAE converges quite slowly. So I kept the lr flat at 0.0005 so I can just continue on the best experiment when trying new ideas. 

### The main magic
The main boost in score comes from augmentations. In order to augment the data, you basically need to transform the target and then do proper wave propagation to generate the corresponding seismogram. A starting code to do that closely enough to how organizers generated the seismograms was shared [here](https://www.kaggle.com/code/manatoyo/improved-vel-to-seis) thanks to @manatoyo and @jaewook704 . But that generation took 3s per target and was not viable to run on the fly while training. So, just for learning purpose I tried speed up the wave propagation as much as possible using GPUs. I spent days for improving the speed. First, I converted the wave propagation to pytorch on GPU and make it run with torch compile. Then I implemented a batched version of it. Finally, I created  a custom CUDA kernel for batched wave propagation. (very good learning for me). That was fast enough to directly use it in my model to recalculate seismograms whenever the target has been augmented. Speed-up compared to the original CPU implementation is about 100x. I used shift, scale, rotate augmentations and an own implementation for shifting intensities. 

### Inference
On the last 2 days, I took the few checkpoints I had (which were run on 0.0005 lr) and added another 150 epochs with cosine schedule, giving more sample weights to difficult classes. I saved test predictions every 3 epochs and took median of the last 10 saved test predictions, to get a single prediction per run. Then simply took mean over the runs. For time reasons I only trained around 6 models spread over 3 of my 8 folds.


## Idle speculation
I asked 2 people for teaming, but was rejected. So I thought, why not try an [idle speculation](https://www.kaggle.com/competitions/expedia-hotel-recommendations/leaderboard), i.e. try to win with a single sub. That was a learning experience of its own, especially under time pressure. You need to manage training as long as possible, but also hedge the risk of failing platforms (compute/ data host plattform  as well as kaggle). So 2 days before end I already had a "backup" submission ready as csv file in case anything crashes. My solution is definitely worse in score because I only used a single submission approach. For example I only realized 2 days before end, that you only need to predict every 2nd pixel. If I realized that earlier I would have been able to train twice as fast. Also how models and folds are selected and weighted was just by gut feeling. Although I see high potential of my solution especially if I would have put any effort in modelling, I am very happy that my submission worked out and I achieved solo gold in this restricted time window. 

Thanks for reading.
