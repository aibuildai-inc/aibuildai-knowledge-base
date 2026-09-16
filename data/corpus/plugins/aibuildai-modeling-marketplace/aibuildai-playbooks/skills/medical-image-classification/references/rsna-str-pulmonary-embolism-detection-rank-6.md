# 6th place solution

Competition: rsna-str-pulmonary-embolism-detection
Rank: #6
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/195865

Congratulations to all the winners. Thank you to Kaggle and RSNA for hosting this competition.
I'm happy about my result because I'm a medical doctor and therefore, RSNA competitions are the most important competitions for me. In RSNA 2018, I did my best but I failed to get gold. I couldn't participate in RSNA 2019 because I had to prepare for the national exam for medical doctors. In RSNA 2020, I've got the solo gold finally. I'm so sad that I can't attend the RSNA conference in place because it becomes an online conference.

Here I describe my solution. The overview is shown below. Actually, there is nothing special; 2D-CNN for image-level feature extraction and 1D-CNN for exam-level classification.

[Figure 1]
# Stage 1: 2D-CNN for feature extraction
First, I trained 2D-CNN (EfficientNet B0 or B2) with trainable 3 windows (WSO, [following Yuval's solution](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/117480)) on 2D images with the condition shown below. To save memory and time, I trained models with mixed-precision.
- Loss: BCE with the weights reflecting the competitions matric weights
- Window: initialized with [Ian's windows](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930)
- Image size: 512 (law image)
- BatchSize: 80 for B0/50 for B2
- Training Steps: 8192 (= 0.5 epoch)
- Optimizer: Adam
- LR: 1e-3 decline to 1e-4 with cosine annealing
- Augmentation: ShiftScaleRotate, BrightnessContrast, Crop (448 x 448), CutOut

Training takes 3 hours x 5 folds with P100. I extracted 2D-CNN's feature, the output of the global average pooling layer, and used it as input for stage 2. Feature extraction takes 2. 5 hours x 5 folds.

# Stage 2: 1D-CNN for exam-level classification
Next, I trained 1D-CNN for exam-level classification. As input, I used feature sequences extracted by 1st-stage 2D-CNN. Feature pooling, Skip connection, and SE-module are employed. For image-level prediction, I used U-Net-like upconv architectures. The training conditions are shown below.
- Exam-level loss: BCE with the weights reflecting the competitions metric weights
- Image-level loss: BCE and BCE with the q_i weights
- BatchSize: 64
- Epoch: 16
- Optimizer: Adam
- LR: 1e-4 decline to 1e-5 with cosine annealing
- Augmentation: Crop (128 slices), Flip
Training takes 6 minutes x 5 fold.

# Postprocessing
I averaged the predictions of B0 and B2. I did postprocessing to solve the conflict of label consistency with minimal modification. The 5-fold CV score is shown below.
```
negative_exam_for_pe           bce: 0.345601, auc: 0.895471
indeterminate                  bce: 0.084615, auc: 0.812207
chronic_pe                     bce: 0.158846, auc: 0.682470
acute_and_chronic_pe           bce: 0.081380, auc: 0.842780
central_pe                     bce: 0.111196, auc: 0.949153
leftsided_pe                   bce: 0.287363, auc: 0.900652
rightsided_pe                  bce: 0.298690, auc: 0.911001
rv_lv_ratio_gte_1              bce: 0.228942, auc: 0.902246
rv_lv_ratio_lt_1               bce: 0.342175, auc: 0.835369
exam-level score               bce: 0.196321
q_i weighted_image_bce         bce: 0.206749, auc: 0.965775
total_score                    bce: 0.201473
```
# Final submission
As final submissions, I selected
1. B0 model with 5-fold averaging: public: 0.161, private: 0.157
2. B0 and B2 model with 5-fold averaging and model averaging: public: 0.160, private: 0.156

# Comparison test
Stage 1
```
                 exam BCE   exam AUC   image BCE   image AUC
B0 final model   0.314768   0.709304   0.101278    0.951922
 num step X2     0.316794   0.708131   0.100873    0.951224
 initial LR=1e-4 0.321354   0.698805   0.113893    0.938055
 BatchSize=16    0.320763   0.687397   0.101027    0.950439
 w/o CutOut      0.316906   0.703339   0.100654    0.950882
 InputSize=256   0.328765   0.681211   0.134463    0.917054
 w/o WSO         0.315638   0.703576   0.103514    0.947023
 with MixUp      0.317532   0.693032   0.103800    0.944957
 InputSize=640   0.321643   0.703420   0.108685    0.953675
B2 final model   0.315517   0.707378   0.099449    0.950943
```
I did comparison tests after the deadline. scores are calculated without any weights. 0.5 epoch is enough to converge. InputSize=512 is better than InputSize=256. InputSize=640 may be better than InputSize=512. LR=1e-3 is better than LR=1e-4. BatchSize=80 may be better than BatchSize=16. Cutout, WSO, and MixUp may not be necessary.


Stage 2
```
                 exam BCE   exam AUC   image BCE   image AUC
B0 final model   0.200486   0.867666   0.222240    0.964814
 w/o pred-2      0.200840   0.867577   0.221665    0.963847
 LSTM            0.203616   0.864214   0.216788    0.963465
 GRU             0.195599   0.873768   0.219781    0.963367
B2 final model   0.196107   0.871995   0.220244    0.962613
```
Scores are calculated with weights of competition metrics. Training with pred-2 may not be necessary. LSTM and GRU may be able to achieve the same performance as CNN.  


What I had to do in this competition was obvious because this competition is very similar to the last year's one and Yuval, the last year's winner, was at the top of the LB. Actually, What I did was just implementing the last year's solution and making a custom loss that minimizes the competition metric directly. It is a very baseline model. I think my solution would be in the middle of silver medals in usual competitions. This competition is a little bit harder than usual competitions, because of a large dataset, complicated metric, short span, and notebook competition. Maybe that's why this baseline model can get gold.

[All the codes are available here.](https://github.com/OsciiArt/Kaggle_RSNA2020_6th_Solution)
