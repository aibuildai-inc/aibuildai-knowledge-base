# 11th Place solution - Team COTS

Competition: tensorflow-great-barrier-reef
Rank: #11
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307718

We are happy to survive in the shakeup and it turns out trusting CV is indeed the key here! It is a great team effort and it has been a great journey to work with @anjum48,  @yamsam ,  @imeintanis and @markunys . With this gold medal, 3 of us ( @imeintanis , @markunys, and me) will become Competition Master, it is just too good to be true 😆

In general, it is very important to observe the video clip with those model predictions, it helps us a lot for observing some improvement areas, like tracking, ensembling, and adding new data. Here are some key points about our solutions:

Best private LB model pipeline:
[pipeline]

1. **Final Submission Model (2 best CV,  1 best CV & LB,  1 best LB)**
    1. 5 YOLO WBF + 1 Cascade RCNN WBF: only used those predictions from RCNN that has >= 0.3 IOU with YOLO prediction in order to correct YOLO's bounding box).  **CV 775 Public LB 626  Private LB 720**
    2. 7 YOLO WBF: **CV 781 Public LB 628  Private 719**
    3. 4 YOLO WBF + 1 best LB model WBF:  4 YOLO WBF CV 771,  **Public LB 653, Private LB 706**
    4. 2 best LB model WBF: Not sure CV,  **Public LB 718, Private LB 666**.
2. **Ensemble, final submission [notebook](https://www.kaggle.com/vincentwang25/11th-place-solution-100-iterations-journey)**
    1. Model inference with 1.3 x training size (multiscale inference in yolov5)
    2. low confidence threshold for every single model and high confidence threshold after WBF  (**~0.005 CV improvement** compared with using high conf threshold before and low conf threshold after WBF)
    3. Model picking criteria:  diversity from data (CLAHE preprocessing or not), training method (1 stage, 2 stages), and model (yolov5 or rcnn)
3. **CV scheme**
    1. manually picked sequence that represents 20% of data (4707 images)  --  balancing between representative and data size. **This results in a final CV and PVT LB correlation of 90%.** 
         1. Pick criteria: # of COTS per frame + how hard it is to predict by f2 score (we don't wanna pick a subset of sequence that is too easy or too hard to predict) + sequence across different video.
    2. Not using 5 folds because of computational constraints.  Not using video_id because it takes too much data away.
4. **Training**
    1. training with all annotated images +  ~5% background image  (those background images are picked from those high FP images or randomly, it **improves CV round 0.005**).
    2. Using CV to choose hyperparameter and best epoch number --> training with all data for submission.
    3. YOLOv5: default hyperparameter (mixup, mosaic, flip, HSV, translate, scale) but with learning rate 0.001 and Adam optimizer.
        1. 1 stage model: training with GT data for 20 ~ 40 epochs, batch size 4, training image size 2400~3600 (depending on model size).
        2. 2 stage model: training with original data like 1 stage model, then with GT data for 10 epochs with 0.0001 LR rate.
        3. Some models also used `--label-smoothing 0.2` for faster convergence and higher CV score.
    4. Cascade-RCNN:  mmdetection framework with augmentation flip, rotate, CLAHE, HueSaturation, RandomBrightnessContrast, RandomSizedBBoxSafeCrop.
5. **Postprocessing**
    1. Tracker for each model before WBF (**~ 0.01 CV improvement** compared with using tracker after WBF)
    2. For the tracker, we filtered out those predictions from trackers that are right on the edge because they are usually FP (this **increases CV by ~0.002**).  Also, we use `initialization_delay=2` instead of 1 to depress too many FP, this **improves CV by ~0.005**.
    3. Using Optical Flow ([notebook](https://www.kaggle.com/yamsam/optical-flow-by-raft)) to reset tracker to reduce FP (~ 0.01 private LB improvement)
4. **Data**
    1. GT data: Add COTS through WBF prediction: Some real COTS is not marked in original data. We added it manually through observing the predicted video clip. In total we added 1213 COTS, around 10% of the original COTS number. This **improves CV ~0.01**.
5. **Things didn't work**
    1. SAHI
    2. modify tracker to reduce obvious FP during each sequence.
    3. Using other datasets, like [DUO](https://github.com/chongweiliu/DUO)
    4. Trying to use best CV model and best LB model together to improve LB score but never get better than original best LB score (due to the tightness theory.....)
    5. YOLOX, YOLOR, YOLOv5-p7, Swin Transformer with Retina, fasterRCNN and TPH YOLOV5.
    6. Tried to add mosaic and mixup in mmdetection framework but didn't work.
6. **Hardware / # of gpu hours**
    1. 3 x RTX 3090 +  2 x 2080Ti + 1~2 GPU with memory 34GB + colab pro
    2. GPU hours 850+ hours

List of final ensemble candidates and their CV score in case someone is interested.
[pic]

In the end, thank you to the great Kaggle community for many amazing sharing throughout the process, it is an exciting learning experience!🔥🔥
