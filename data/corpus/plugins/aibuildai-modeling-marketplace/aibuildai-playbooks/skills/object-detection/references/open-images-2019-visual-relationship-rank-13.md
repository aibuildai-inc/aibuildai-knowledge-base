# 13th place solution summary

Competition: open-images-2019-visual-relationship
Rank: #13
Source: https://www.kaggle.com/c/open-images-2019-visual-relationship/discussion/110935

Hi all. Conglatulations to the winners and thank you Google AI to host this interesting competition this year again.

Like most of previous years solutions, I split the problem into two parts, `non-is` and `is` relationships as they have quite different characteristics.

### 1. non-is relationship

For this task, I focused on the relationship between two objects such as `Man on Chair`, `Cat under Table`. 
My approach has two steps. Detect objects and then find out relationships for every possible triplets.

**Object Detection**

There are 57 objects such as `Man`, `Oven` to be a part of triplets. I used cascade-rcnn from [mmdet](https://github.com/open-mmlab/mmdetection/tree/master/mmdet) for these 57 classes with a bit of modification such as adding test time augumentations.

[This](https://github.com/appian42/kaggle/blob/master/openimages/cascade-rcnn.conf) is the .conf file I feeded to mmdet. Noticeble changes from default parametersz are 

- 2 more anchor boxes in RPN to capture high aspect ratio objects.
- CosineAnnealing instead of StepLR.
- score threshold 0.0001 for RCNN instead of 0.05.
- NMS treshold 0.4 instead of 0.5.
- max\_per\_image 400 instead of 100.

I under-sampled frequent classes such as Man, Woman, Chair and used only 150,000 images to shorten the train time at the cost of accuracy.

**triplet relationships**

There are 287 triplet relationships.
I took a similar approach to [anokas's solution](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64630) last year as it's extremely simple and easy to implement.

Some changes I made was

- Merged similar triplets into the same class based on some engineered features such as IOU, IOF so that rare triplets can be trained thanks to more frequent triplets.
- After merging, there are 90 classes out of 287 triplet relationships and 4-fold LightGBM models were separetely trained for each of these classes.

Averaged AUC is 0.9641. After separating merged classes into original classes, averaged AUC is 0.9623.


**Submission**

- 0.31692 (public)
- 0.24060 (private)


### 2. is relationship

**Object detection**

I used cascade-rcnn to directly detect attributed objects such as `Table Wooden`, `Bench Plastic`. There are 42 classes for this task. This approach is similar to [toshif's solution](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64642) last year but I failed to make the model as good as he did. 

**Submission**

- 0.07346 (public)
- 0.07130 (private)


### 3. submissions combined

I just put them together.

- 0.38469 (public)
- 0.30781 (private)


### 4. Possible improvements

- Use full image size.
- Use all dataset.
- Use external dataset (such as COCO, Objects365 if it improves any).
- Train CNN model for triplet relationships just like [tito's solution](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track/discussion/64651) last year. Some triplet relationships such as `Man holds Violin`, `Man plays Violin` are hard to differentiate because simple features such as IoU, IoF could not really capture the difference but CNN could.


Thanks for reading!
