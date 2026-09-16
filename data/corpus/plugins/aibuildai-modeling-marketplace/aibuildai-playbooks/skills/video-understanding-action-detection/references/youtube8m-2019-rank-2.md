# 2nd Place Solution

Competition: youtube8m-2019
Rank: #2
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/113663

Congrats to all the winners and thanks google research for providing such an interesting video understanding challenge. 
I would like to share my solution here. Scores shown in the following tables are all private scores, including the private MAPs, which are about 0.01 lower than scores on the public leaderboard.

# Solution Overview

We firstly pre-train base models using videos from 2018-data, then those models are fine-tuned on all the segments(2019 data) using a segment-level loss. Finally, a refinement inference strategy takes in both the video-level predictions and the segment-level predictions to obtain the refined segment-level predictions.

## Video-level Pre-train
We combine the mixture structure (3rd place last year) with different video classification models.
**Frame-level Model**: Mix-[NeXtVLAD, nonlocal-LightNetVLAD, nonlocal-EarlyNetVLAD, GatedDBOF, SoftDBOF, NetFV, GRU].

**Video-level Model**: Mix[ResNetLike]

We train models using the 2018 large-scale YouTube-8M video-level annotation data. The trained models were evaluated on last year's video label prediction task. As shown in the table below, most frame-level models with the mixture structure can achieve high scores on the leaderboard, and the simple equivalent weight ensemble of these models can reach 0.88932 on the private GAP which is a high score (although we do not consider the model size limitation).  These single models are regarded as ``Base Models" and were evaluated on this year's temporal localization task. Their  MAP scores are shown in the second column. The Mixture-SoftDBOF achieves the best MAP, followed by the Mixture-GatedDBOF and the Mixture NeXtVLAD model.



## Segment-level Fine-tune

**Fine-tune Loss**: For a certain segment, suppose A is the set of 1000 segment-level categories, B is the annotated segment category, and C is the set of annotated video-level categories of the video which the segment located in. Cross Entropy Function represents as CE. We add some weak supervision information into our final loss function shows below.   *α* is simply set as 1.0 in our experiments.
\\( Loss = \Sigma\_{i\in B} CE(p(i), L(i))  + \alpha * \frac{\Sigma\_{i\in (A \setminus C)} CE(p(i), 0) }{|A \setminus C| }  \\)

**All Data**: In order to utilize all the annotation data, we set two groups of experiments for each model. We first use 5/6 segment data as our training set and 1/6 data as the validation set to train models. Through the validation results, a good model step interval can be estimated when using full data as the training set. Finally we use the Stochastic Weight Averaging (SWA) technique to combine those models into a single one. On the one hand, training step interval estimation improves the tolerance of the model selection. On the other hand, SWA operation improves the robustness of the models and can gain higher scores.

## Refinement Inference Strategy


### Basic Inference Strategy
The basic version of the inference method creates 1000 minimum heaps for segment-level predictions. The predictions for each segment are pushed into the heap of respective categories. Once a heap size overflows the maximum threshold, the segment with the least predicted probability in the heap will be popped. Finally, the segment classification predictions can be converted into the final temporal localization results by sorting confidences in the heap.

### Refinement Inference Strategy
The basic inference strategy ignores the powerful instruction of global video information. In this part, we will utilize video-level predictions to improve the segment-level predictions. 

Considering the fact that if an entity has very low confidence in appearing in a video, then it is also unlikely to appear in any of the segments in this video. Based on this consistency observation, we build a list of candidate labels for segments classification. The list of candidate labels is obtained from video labels that predicted by pretrained models, and is quite effective in removing false positive predictions on video segments. 

**Top rank k**

Our first thought is to select the top k predicting results on the video data. These selected classes consist of candidate categories, and are used to constrain the probable category scope for each segment.

**Confidence threshold to constrain**

The top k strategy is a good way to generate filters, but it ignores the diversity of categories between videos. For content-rich videos, it contains a large number of categories, while for a single-content video, the number of categories appears will be small. 

So the main idea of our second strategy is to consider the confidence. If the prediction score is smaller than the threshold, then the related category won't be considered when predicting the inside segments.

**Video number constraint for each category**

The third idea is to simply limit the number of predictable videos for each category. We predict each video using video-level model, and then limits the predicted video number by confidence for each category. Only segments in these videos will be considered as the corresponding categories.



## Ensemble

Finally, ensemble models can achieve great improvement. We finally choose Mix-NeXtVLAD, Mix-GatedDBOF, Mix-SoftDBOF, Mix-EarlyNetVLAD, and three kinds of Mix-ResNetLike Models for the final ensemble.  Ratios we set are 3:3:3:1:1:1:1. More models for ensemble and other ensemble ratios may bring some improvement to the final MAP scores.





##  Other things we tried but did not work so well

1. auto-annotation to achieve data augmentation
2.  reranking by combining video-level and segment-level predicted confidence
3. fix params: fix some parameters to prevent over-fitting
