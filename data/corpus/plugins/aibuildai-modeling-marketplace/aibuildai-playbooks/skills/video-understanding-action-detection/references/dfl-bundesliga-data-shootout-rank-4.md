# Public 4th place solution

Competition: dfl-bundesliga-data-shootout
Rank: #4
Source: https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/360573

First of all, thanks to organizers for their effort in this amazing competition and congrats to all the winners. Although we need to wait another two months for the final results, let me share my currently 4th place solution. 

# Overview
The basic pipeline of my solution and the summary of the models are shown below.

| Block | Model | Backbone | Input | Output |
| --- | --- | --- | --- |
| Frame Extraction | NA | NA | 1920x1080 | 960x540 (odd frame only) |
| Candidate Frame Selection | Image Classification | efficientnetv2_l | 512x512 | 4-class probability |
| Event Area Detection | UNet | efficientnet_b5 | 512x512 | Heatmap |
| Event Classification | [CSN](https://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Video_Classification_With_Channel-Separated_Convolutional_Networks_ICCV_2019_paper.html) | ResNet152 | 224x224 ROI | 4-class probability |

# Frame Extraction
I used FFmpeg to extract and save frames in a video. Note, to reduce the time for disk I/O, I decided to discard even frames and reduce the size of frame to 960x540 (i.e., half the original size).

# Candidate Frame Selection
The purpose of this step is to find the candidate frames in which events are likely to occur. I simply classified each of the extracted frames to four classes, background (not event), play, challenge, and throwin. The backbone is [timm](https://github.com/rwightman/pytorch-image-models)'s efficientnetv2_l, and an input frame is resized to 512x512. 

This process doesn't use temporal information at all, so the accuracy itself is not high, but what I wanted to do here is to find the event frames with a high recall. So I ignored the absolute values of classification scores, and focused on the relative peaks. I used [scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html), and picked the frames corresponding to the peaks of the classification scores as candidate frames.

# Event Area Detection
Through a set of experiments, I found that it's difficult to reach a competitive score if I naively used entire frames. Since [hand-labeling is allowed in this competition](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/342948), I decided to train an event area detection model to extract a ROI from frame. I annotated xy coordinate of (roughly) center of an event area (basically the coordinate of player who is most closely associated with the event) for all the event frames in the training set. This took several weeks :-(

The event area detector is a simple UNet, and it outputs a heatmap of event area. The backbone is [timm](https://github.com/rwightman/pytorch-image-models)'s efficientnet_b5, and an input frame is resized to 512x512. Some example results are shown in the figure below. In the figure, the left is input, the middle is output heatmap overlapped to the input, and the right is zoom-in image centered at the peak of the heatmap.

To detect a ball is one of the promising approaches in this competition, but as discussed [here](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/343652), in many cases a ball is not visible (as you can see from the examples above), or is quite difficult to detect because of various reasons such as background color, lack of resolution and motion blur. So I decided to detect an event area instead of a ball.

# Event Classification
The frames from the **Candidate Frame Selection** contains a lot of wrong (non-event) frames, and also the scores are not reliable. To refine the results, this process applies a more sophisticated action recognition model to the candidate frames. 

After trying several action recognition models, I employed [3D Channel-Separated Networks (CSN)](https://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Video_Classification_With_Channel-Separated_Convolutional_Networks_ICCV_2019_paper.html) since it showed good performance for both accuracy and speed. The training set for this model was created based on the outputs of two previous steps, **Candidate Frame Selection** and **Event Area Detection**.  First, I applied the candidate frame selection to all the frames in the training set, and set the labels to background for the wrongly selected non-event frames. Of course, the correctly selected frames had their own events' labels. Then, I used the event area detection model to extract a ROI from each frame. The center of ROI is the peak of output heatmap, and its size is 224x224. Since the number of background frames are much larger (x~10) than those of event frames, I randomly selected around 10% of background frames every 5 epochs during training. 

The backbone of CSN is ResNet125, and the number of input frames is 16 (actual time span is 32 frames because even frames are discarded). At both training and inference, a ROI is extracted from candidate frame, and the same regions are also extracted from +/- 8 frames, and fed to the CSN.

The final score is just a probability output from CSN, or ensemble of CSN output and the score from **Candidate Frame Selection**. The latter is slightly better than the former (+0.01).  

# Computation Time
Since my pipeline consists of several blocks, my biggest challenge was to keep the computation time for each block as low as possible. Using only odd frames is one of the biggest compromises. The per-video computation time for each block in the kaggle kernel is something like below.
| Block | Computation Time |
| --- | --- |
| Frame Extraction | 15 min |
| Candidate Frame Selection | 20 min |
| Event Area Detection | 15 min |
| Event Classification | 30 min |
| Total | 80 min |

I should have thought about simpler end-to-end method, but I couldn't. Now I'm really impressed by other kagglers' very sophisticated methods, and learning a lot. Thank you, everyone!
