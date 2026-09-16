# 4th Place Solution - CenterNet

Competition: tensorflow-great-barrier-reef
Rank: #4
Source: https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/307626

Since overfitting strategy failed, let's just talk about what I've done one month ago. 😅

I use CenterNet with DeepLabV3+ architecture and EfficientNetV2 backbone, start from this [example](https://keras.io/examples/vision/deeplabv3_plus/).
- change backbone to EfficientNetV2 B0~XL
- change output heatmap size to 1/8 input, and add regression head
- training on 1280x720, inference on 1792x1008(1.4x)
- blend two can get private LB score 0.712, and 0.722 if inference on 1.6x

Actually, I am pleased to survive in huge shake up. 😃

Thanks everyone and congratulation to all winners.
