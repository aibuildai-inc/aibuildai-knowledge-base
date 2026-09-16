# 15th place solution: YOLO-X only, Seq-NMS

Competition: tensorflow-great-barrier-reef
Rank: #15
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307691

Congratulations to the winners and all participants who finished this competition 😊
In this competition, many participants were focused on increasing the score of Public LB, and I was not able to keep up with Public LB at least halfway through.
However, in my local experiments, I have found that methods that increase the image size only during inference, even when the threshold and ensemble methods are properly adjusted, result in extremely low scores for the entire training data.
So, I thought there would be shake. Although I was not able to get into the gold zone, I am glad that I participated because I was able to recognize the necessity of believing in my own experiments.

Ok, my solution is a pretty simple one, but I would like to share.


**1. Preprocessing**
Since the video was basically switching almost every sequence, I built a validation with 4 folds based on the sequence (However, I think there were some sequences that were a little continuous).
One coin I did was to make the number of CoTS almost the same in each fold. In training, I did not use images without CoTS, but in validation, I used all images without CoTS in order to check the False Positive properly.
  The image was simply divided by 255 without any noise removal, and the size was 1952 x 3520.
Due to GPU limitations, I couldn't try a larger size, but at least up to this size, I was able to get a small gain with Local CV. However I got the most gain up to about twice the size of the original image.


**2. Model: All I want to use is YOLO-X**
This time I really wanted to use `anchor-free` YOLO-X, which I had never used before, so I stuck to it, even though many people have had success with YOLO-v5 in Public LB.
  It was hard to modify YOLO-X because many parts were hard-coded, but I tried to tweak the [top-K selection algorithm](https://arxiv.org/pdf/2107.08430.pdf), add augmentation, and some other dubious modifications. But in the end, only augmentation seemed to have an effect. In that sense, YOLO-X's perfection as a model may be high 😏


**3. Augmentation and learning strategy**
In addition to the augmentation (mosaic, mixup, etc.) that YOLO-X has as a default function, I added RandomGamma, RGBshift, Sharpen, GaussNoise, etc.
Instead, the probability of applying mixup and mosaic has been slightly reduced from the default value and assigned the rest probability to the newly added augmentation path. This may have increased the diversity of the input images somewhat, and I was able to get a gain in Local CV(I don't know the specific ablation values in detail, as the experiment took some twists and turns).
  In addition, I used the [progressive learning](https://arxiv.org/abs/2104.00298) method used in EfficientNetV2: gradually increasing the size of the image (e.g. 1280 => ... => 3520) as the learning progressed. At the same time, I remember that regularization (increasing the probability of application of augmentation) was also strengthened somewhat.


**4. Inference**
The inference was very simple: I did TTA with Flip and ensembled with WBF for 4-fold (so 8 models). I think this is almost same as many of the participants.
Note that I set thresholds to improve the local CV. The F2 metrics were sensitive to the threshold settings to some extent because our evaluation metric was based on the confusion matrix.


**5. An original point that I have worked out: Seq-NMS**
One point that I devised a little is the post-processing.
  Since the task is object recognition in video, I made a post in [this thread](https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/293812) early stage in this competition, and I was reading almost all papers. Many of them reported that feature aggregation, which is a method of enriching input features by using past images in the neighborhood, is more performing than post-processing methods such as tracking, and I thought that was probably true. However, since the majority of feature aggregation methods is based on RPNs, it was too much of a hurdle for me to apply, since I was sticking to YOLO-X, which is anchor-free. So I decided to use [Seq-NMS](https://arxiv.org/abs/1602.08465): a typical post-processing method. Specifically, the confidence is adjusted by the degree of overlap between the predicted bbox of the previous images and the predicted bbox of the current image (Ref: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/293812#1611428).
(The implementation can be found [here](https://github.com/tmoopenn/seq-nms). The core processing is written in cython and is fast enough, but as far as I understand, there is a serious bug in the python script that makes it not work properly as is, so I fixed the bug to use it.)
  As long as I experimented, there is no need to consider many frames for Seq-NMS, just the prediction of the previous image. I also tried to use [Dense Optical Flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html#:~:text=Dense%20Optical%20Flow%20in%20OpenCV) to compensate for the shift of predicted bboxes of the previous frame, but in the end I did not use it because Dense Optical Flow is expensive to compute and does not have much gain.

---

That's all I can remember right now. If I remember anything else, I'll add it.
Thank you all for your hard work, and see you at another competition :)

Happy Kaggling😊




Feb. 18. 2022, Model pipeline figure updated
