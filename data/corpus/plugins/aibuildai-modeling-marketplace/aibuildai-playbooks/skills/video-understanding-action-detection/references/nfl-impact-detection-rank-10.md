# [10th place solution] Extended Centernet

Competition: nfl-impact-detection
Rank: #10
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/208773

Thanks to NFL and Kaggle for hosting this interesting competition. Congratulation to all, especially to the prize winner and gold medalists.

*The following is my approach.*

**1. Detection models**
My detection model is derived from [Centernet](https://github.com/xingyizhou/CenterNet) (MIT licenced) with some modification.
- I replace the image-based feature extraction by a video-based feature extraction. Getting video features is very useful for deteting impact.
- I stack multiple centernet's heads on the top of the feature extraction block. Each head is responsible for predicting helmet for each frame.
- I calculate loss independently between 2 classes (helmet with impact and helmet without impact) and then using weighted sum of the 2 losses. 
- I write my own augmentation for video.


- Input: I use a sequence of 15 consecutive frames. I also use a customized sampling method that focus more on frames with impact. I think using longer sequence can improve the result further, but my GPU does not allowed me to do so.
- Video feature extraction: I use EfficientB5 (need to modify number of input channels). I've tested EfficientB3, EfficientB5 and EfficientB6. I don't have enough time and GPU to explore further.
- Detection head: I've used only Centernet for this detection head, because I am more familiar with it.

**2. Post-Processing**
- IOU-Tracker: I simply use [IOU-Tracker]( https://github.com/bochinski/iou-tracker) (MIT licenced) to link the detected impacts from near by frames. For each tracked trajectory, I just keep the middle box and ignore the rest.
- Dynamic confidence threshold: Since the impact most likely to happen from frame 30th to 80th, I use a low threshold within this range. The number of impacts decrease to the end of the video, so I slowly increase the confidence threshold to the end of the video. Below is the threshold I use for the whole video

