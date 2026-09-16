# 15th place solution: single stage object tracking

Competition: nfl-impact-detection
Rank: #15
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208918

FIrst of all, thanks to Kaggle and host for having such a interesting competition. And also congrats to all the winners. I believe it was one of the toughest competition since there were so many talented kagglers while total number of competitor was relatively small.

Here I'll write up a short summary of my solution and also where I failed.

# TLDR;
- No ensemble single stage model
- FairMOT as baseline for helmet tracking
- Additional impact detection branch and team feature branch
- A bunch of postprocess with track id, team and position


# Baseline

At first sight this task seems the best application of object tracking because we need to aggregate timeseries information due to metric design. So I chose [FairMOT](https://arxiv.org/abs/2004.01888) as my baseline because it is the SoTA of the MOT benchmark. In short, FairMOT is object tracking framework which is CenterNet with metric learning based id feature branch. Additional to helmet tracking, my model also predict impact probability for each impact types and also team feature to distinguish opponent helmet. So in total, my model's outputs are following:
- Helmet image coords and bbox wh and confidence (N * 5)
- Id feature (N * 128)
- Team feature (N * 128)
- Impact probability (N * 6)

where N is number of helmets.

As a small trick, I added same tracking label for same player in both view, so my model should output same id feature for same player in both view. While tracking I associated helmets in both view. Additionally I devide helmets in 2 groups, or teams, in the view. I simply use kmeans to clustering, but it works very well. After tracking, My model's outputs are following (in addition to above):
- Tracking id (N,)
- Matched tracking id in another view (N,)
- team class (0 or 1) (N,)

Other small details of training:
- backbone: HRNet32
- image size: (708, 1280)
- num epochs: 20 for Endzone, 35 for Sideline
- extend impact label to +- 4 frames

# Post process
Even though my model predict impact probability for all impact type, in my best submission I used only prediction for impact between helmets. And my postprocess procedure heavily rely on the hypothesis in which helmet impact happens between the pair of helmets from different team.
My post process is consists of following steps:
1. Calculate maximum iou with opponent helmet
2. Take maximum of impact probability between the pair of highest iou helmets
3. Calculate minimum pixel distance with opponent helmet and normalize it by mean box size in frame
4. filter by helmet detection confidence threshold
5. filter by distance threshold
6. filter by impact probability threshold
7. maxpooling nms in time axis for same id
8. remove solo helmets which has no pair helmets in near frames

As you can see there are a lot of hyper parameters. I optimized these thresholds with optuna for Endzone and Sideline separately. 
By doing so my cv was about 0.42 for Endzone and 0.53 for Sideline, and 0.45 in total. CV and LB correlation was not so bad, my final score was actually 0.45 both on public and private.

# Where I failed?
Obviously I missed to aggregate timeseries information, such as 3d model or 2 stage model which is most of other solutions have. Instead of that, I spent most of time in last month to aggregate relationship between instances in same frame and other view. I believe there shuold be some breakthrough but I couldn't make it works. What I tried were:
- Making use of other view prediction. (ex. if there is no prediction in other view, filter out more)
- Making use of matching id even though it has best matching accuracy among the method I tried.
- Homography estimation and optimization between End and Side or image and sensor tracking position.
- Training and matching with [superglue](https://github.com/magicleap/SuperGluePretrainedNetwork). I used id feature as descriptor and center points as keypoint. I added superglue on top of FairMOT and trained simultaneously. It works somehow but simple linear sum assignment was better than this.

I'd like to release code, but it's so messy now. If I success to clean it up, I'll update.
