# [40th place solution] 2-Class Prediction + HFlip + NMS + Filtering

Competition: nfl-impact-detection
Rank: #40
Source: https://www.kaggle.com/c/nfl-impact-detection/discussion/209341

Firstly, thanks both to Kaggle founders and the NFL competition organisers for the possibility to take part in this contest! Secondly, thanks to @its7171 and his [kernel](https://www.kaggle.com/its7171/2class-object-detection-training) for the great idea of 2-class training!
# Brief Summary
The [Tiny-YOLOv4](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4-tiny-3l.cfg) from[ AlexeyAB's darknet repository](https://github.com/AlexeyAB/darknet) was used. The training of both impacts and non-impacts with leaving impacts only was done. To gain more accuracy, horizontal flipping data augmentation both when training and testing, filtering and non-max suppression of both (flipped and original) frames were implemented.
[The complete pipeline]
# Training
The only videos and train_labels.csv were used in training. No separate frames from image_labels.csv and tracking data were used because of the lack of time and resources.

The parameters are:
- batch_size: 64,
- classes: 2,
- iterations: 4000 (with 64 images per iteration),
- step: 15,
- augmentation: horizontal flip only,
- total images: ~4000 (2000 * 2),
- input size: 896x512 (with random resize).

It was used the step to extract frames from videostream to decrease the consumed time of training and also to make difference between impacted and non-impacted helmets.

# Inference

The main idea is to use not only one but two images per the real video frame: the original and the horizontally flipped one.

When the prediction of both them is done, it's made the filtering by following parameters:
- nearest_index_distance: 5 (the farthest distance between two nearest impact frames),
- farthest_index_distance: 23 (the distance between start and stop frames of the filtered sequence),
- nearest_iou: 0.4 (the Intersection-Over-Union metric).

When the filtering is done, it's made the Non-Max-Suppression to merge the overlapped bboxes and exclude the non-overlapped ones from the submission. The NMS is done on not the only one frame but few nearest frames in the sequence. So, the parameters are:
- nms_max_distance: 4,
- nearest_iou: 0.4.
