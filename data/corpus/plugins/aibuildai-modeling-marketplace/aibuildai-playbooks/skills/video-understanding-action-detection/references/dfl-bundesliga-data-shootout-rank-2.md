# 2nd Place Solution

Competition: dfl-bundesliga-data-shootout
Rank: #2
Source: https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/360097

Thanks to organizers for this interesting challenge and congrats everyone who enjoyed it! Since this competition can be approached in a variety of ways, I'm looking forward to see everyone's solution.

# Overview of my solution
My pipeline consists of:
- **Optical Flow to get global(camera) motion.**
- **Ball Detector to find a ball on the field.**
- **Cost Minimization of Ball Trajectory to refine the ball locations over multiple frames.**
- **2-stage Action Recognition to classify the events.**
- **Ensemble and post-processing.**

Once I found that the attention or cropping near the ball is effective to improve the accuracy of action classifier, I focused on refining the performance of the detector and classifier. 

[pipeline]

---

# 1. Optical Flow
It's difficult to detect the ball when:
- it is occluded behind players.
- its background is the stand.
- balls not on play are placed near sideline.

So my idea is to use multiple frames to detect the ball on play which is usually moving by the optical flow.
At first, I implemented the optical flow with high resolution image since balls are very small, but it's slow and not suitable for this competition. So I adopted the low resolution OpticalFlow to predict only the global (camera) motion. Flow itself cannot tell which one is the ball on play, but the difference between warped image and original image is very informative.
[opticalflow_motivation]
I used RAFT to predict optical flow. It is trained with the provided video by self-supervised manner.
[opticalflow_implementation]

---

# 2. Ball Detection
To make the ball detector, I repeated training and annotation several times. 
(Annotation -> Train Detector -> Predict -> Annotation -> Train Detector -> Predict -> …)
I selected CenterNet because I do not like annotation. Annotating point is much easier than rectangle. 

Since the target(ball) is very small. Inputs to the model is high resolution images.
To achieve both inference speed and accuracy, I used the initial part of the encoder. 
[balldetector]

Inputs to the detector consist of 9 channels:
- normal RGB image of the current frame
- difference between the current frame and previous frame reconstructed by the optical flow.
- difference between the current frame and next frame reconstructed by the optical flow.

The latter two drastically improved the accuracy of detector especially in the crowded scene.
[detection_example]

---

# 3. Cost Minimization of Ball Trajectory
Ball location can be estimated more accurately by using multi-frame information. I applied cost minimization on selecting the ball path. 
If we know the trajectory of the ball, we can estimate where it will go next. In the following case, people can select the next point correctly even if the ball is hard to seen.
[ball_path]

As I want to consider both detector's score and distance based probability, I defined 
  Cost = Function (Confidence of Detector, Distance between Nodes)
and minimized the total cost of the path by Dijkstra Algorithm .

It can create a more natural and accurate trajectory than just connecting points with the highest scores.
[ball_path_image]
(If this competition was for the ball detection, I would get first place!)

Videos can be seen here. Left one is the max confidence path and right one is the min cost path.
https://twitter.com/i/status/1582747059165495297

---

# 4. Event Classifier (Action Recognition)
Now we know where the ball is. Next step is to predict event. I used ball trajectory and images cropped near ball over 1 sec to predict action(4 class) and regression of residuals.
Since the labeled intervals are very short, it's difficult to make the long sequential training data. I extracted only 1 sec frames randomly from labeled intervals as an input to the model.

The model consists of :
1) 2D CNN to extract features from cropped images
2) 1D CNN to extract features from ball path
3) 1D CNN to predict event from upper two sets of features 
[event_detect_1st]

---

# 5. Event Classifier [2nd Stage]
The first classifier cannot predict the event with long-term information.
At second stage, long sequential inputs(4 sec) can be adopted because the inputs of the model is only feature vectors which is much more memory-efficient than the images.
[second_stage]

---

# 6. PostProcessing
Simply, combination of gaussian weight and peak detection.
- Peak detection by max pooling
- Gaussian filter on the “challenge” score because the peak of “challenge” is ambiguous compared to the other two classes.
[post_process]

---

# Final Thought
Since the number of samples of the events is not so large and the most of the area in a image is useless, I thought the model would hard to learn and would overfit easily. So I soon decided to use the ball trajectory.
That's why I really amazed that many kagglers suceeded the simple end-to-end approach. I've learned a lot and still need to learn a lot from this competition. Thanks again to kaggle community!
