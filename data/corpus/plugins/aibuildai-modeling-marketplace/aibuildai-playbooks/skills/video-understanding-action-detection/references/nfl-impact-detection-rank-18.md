# 18th Solution : YoloV5 and filtering using tracking data

Competition: nfl-impact-detection
Rank: #18
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208929

My teammates (@mathurinache  & @matthieuplante) and I would like to thanks Kaggle and the hosts for the organisation of this competition. 
We would like also to thanks @its7171 for the public baseline notebook he provided in the early stage of the competition.

We will describe our solution in an iterative way to see how, from this public baseline, we reached a private LB of 0.42+.

**Public baseline improvement**
We take the idea shared by @its7171 to extend impacts on a [-4, 4] range. But, instead of only training on the frames with impact(s), we added in our train set the frames corresponding to -5, +5 offsets. From our point of view, it helps the model to understand the boundaries between impact or no impact. Using EfficientDetB5 trained on imgsize of 512x512 with a simple 80/20 split, we get as CV/Public/private results : 0.18/0.22/0.18

Based on the previous model, adding post processing that will be describe in last section, we reached CV/Public/private results of 0.22/0.25/0.22

Moving, then, to yolov5 as based model and using original resolution for our 2 classes detection problem, we reached CV/Public/private results of 0.25/0.28/0.24.

Finally, doing some ensembling (that revealed to be bad for private LB) with 0.226 public notebook, we reached a public LB of 0.35+ but a private LB of 0.26+

**Using tracking data as a frame selector**
Tracking data seems a nice playground for us to detect frames with impact as information such as speed and acceleration at player level was provided.

To work with this file, we decided to change a bit it structure from :
[time, player info]

to :
[time, player info, closest opponent info, global aggregates, impact]

Global aggregates are mean/std of player speed/acceleration/… at a given timestamp.
We also take the care to remove any translation or rotation dependent features such as : x, y, direction, orientation. For direction and orientation, we add the scalar product between player and opponent values.
Impact is obtained by reconciliation with the train csv file.

Then, we build a model based on a simple Transformer encoder architecture (6 attentions, 8 heads) with a linear head classifier that will process our new tracking data over 9 consecutive timesteps (9 is a parameter).

We performed a 5 folds training on this data, managing class imbalance by downsampling. Each fold provided more or less a F1 score of 0.4 for more or less 0.4 precision and recall. At first, this seemed poor, but because we are at player level, going back to frame level, we get the following results : recall of 0.7 for a precision of 0.1. Accepting to set an upper bound of 0.7 to our recall, we were now in a situation where 1 frame out of 10 contains at least an impact (original situation on train data was 3 out of 100).

Retraining our yolo based model using an 5 folds strategy (not using the same seed as the frame selector) and using this frame selector for filtering did not improved our result at first ! 
Indeed, as we changed our « prediction space », we needed to change also our validation space by limiting it to the scope of the frame selector.

This strategy gives us a CV/public/private scores of 0.3711/0.3749/0.3767

At last, we enhanced our train set adding the FP from our frame selector (guessing there are some hard samples in it) and we reached the following CV/Public/private of : 0.4117/0.3636/0.4220
Little tweak attempts on postprocessing give us our best public LB of 0.4666 for a private score of 0.4244.

**Post processing**
Several steps composed our post processing :
1 - basic model confidence filter
2 - 5-85% filter : exclude all frames below 5% or above 85% of the video
3 - endzone - sideline constraint : keep only frames where there are predictions on sideline and endzone with at most 3 frames margin
4 - NMS (using torch implementation) on a 9 frames sliding windows with a 0.2 iou threshold
5 - box limit : take only the 4 most confident predictions per frame
