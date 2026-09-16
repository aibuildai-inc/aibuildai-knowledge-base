# 22th Place Solution - single yolov8m with Pseudo label

Competition: byu-locating-bacterial-flagellar-motors-2025
Rank: #21
Source: https://www.kaggle.com/c/byu-locating-bacterial-flagellar-motors-2025/discussion/583289

### Introduction 
First I want to thank Kaggle and the organizers of the competition for the baseline provided. 

Our final solution is a simple YOLOv8m model, the Place Solution—YOLOv8m. 

We experimented with multiple image sizes (640, 960, 1024) and YOLO versions, and we finally decided to use YOLOv8m. We adopted several data augmentation techniques shared by @hengck23 , such as mosaic, mixup, perspective, shear, and auto augmentation. On top of that, CLAHE enhances local contrast and emphasizes fine-grained textures, thereby improving feature discrimination in low-contrast scenarios.

Here's the full setting of our augmentations:
```python
T = [
         A.RandomBrightnessContrast(p=0.2, brightness_limit=0.1, contrast_limit=0.1),            
         A.RandomGamma(p=0.2, gamma_limit=(90, 110)),
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.2),
         A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.3),
         A.ImageCompression(quality_lower=90, quality_upper=100, p=0.1),
         A.CLAHE(p=0.1, clip_limit=2.0),
         A.ToGray(p=0.05),
       ]
```

Our final solution employs a single YOLOv8m model without any ensemble, resulting in a relatively simple inference pipeline. However, we believe that this competition requires participants to focus on in-depth data analysis, particularly in identifying and addressing domain shifts arising from different instruments and acquisition devices. Moreover, it is crucial to design solutions that are robust to threshold variations in order to mitigate the risk of performance degradation on the private leaderboard.

According to the competition's evaluation metric, detections within a relatively large radius are considered correct, and the F𝛽 score used in the competition tolerates a higher number of false positives. This design choice is of particular significance and warrants deeper reflection.

 In the context of medical image object detection, missing true targets is often more detrimental than detecting spurious ones. Models in such applications serve the purpose of narrowing down massive datasets into candidate regions that can be manually reviewed and annotated with higher precision. Ultimately, this human-in-the-loop approach ensures that the final annotations are more accurate. From the perspective of medical practitioners, especially medical students involved in annotation, it is highly desirable for models to achieve high recall while maintaining a reasonably acceptable precision. Such a balance reduces their workload by minimizing missed detections while keeping the number of false alarms manageable.

The key strategy come up from the simple fact I observed. We can iteratively use pseudo labelling to address the missing gt labels in the training data (I think it may be somehow just like the competition organizers they would do on the leaderborad data) . This is also the key reason my 2 yolov8m models can reach plb840 and lb822 (we selected a plb830;lb824 a single model though), which is possible to reach gold zone with only simple yolov8m model!! [840PLB model](https://www.kaggle.com/code/shanzhong8/22nd-solution-tta-yolo-ensemblemodel-plb840/edit)



### **Key Improvements**

* **Correction of data generation issues in the public notebook**:
  We identified and resolved a problem where, in images containing multiple motors, the dataset generation process saved multiple YOLO-format annotations and corresponding `.txt` labels. This led to conflicting positive and negative samples for the same input during training, causing the model to become confused and prone to overfitting to noise.

* **Incorporation of negative samples**:
  We added negative samples to balance the positive-to-negative sample ratio, which helps the model learn more robust decision boundaries.

* **Loss function enhancements**:
  We introduced additional loss functions including **Normalized Wasserstein Distance (NWD) Loss** and **IoU-Shape Loss** to improve the model’s ability to detect small objects effectively.

* **Utilization of additional annotated data**:
  We leveraged 36-box additional data shared by @brendanartley and a subset of 24-box training data. Despite the higher noise level in this supplementary dataset, it contributed positively to model generalization.

* **TTA**:
  Rotation during Inference makes stable performance.

### **Limitations and Unresolved Issues**

* **Lack of ensemble strategy**:
  Due to time constraints, we did not implement an ensemble strategy. Our final submission was based on a single YOLOv8m model.

* **2.5D DEiM model underperformance**:
  Although we experimented with a 2.5D DEiM variant that achieved nearly perfect recall (\~1.0), its performance on the leaderboard was suboptimal, potentially due to a high false positive rate or limited generalization across test domains.


### Summary
Despite the inherent variability in leaderboard rankings, innovative techniques—such as the rank-based thresholding approach in Bartley’s solution—demonstrate how thoughtful design can yield robust results. I am truly grateful for the insights and knowledge shared by the community throughout this competition.
