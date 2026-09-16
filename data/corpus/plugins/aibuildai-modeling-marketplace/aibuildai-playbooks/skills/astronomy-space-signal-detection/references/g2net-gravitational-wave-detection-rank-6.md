# 6th place solution

Competition: g2net-gravitational-wave-detection
Rank: #6
Source: https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/275343

# model architecture
wave -> dct -> trainable bp filter -> idct -> 1dcnn/cwt -> 1dcnn/2dcnn/resnet/effnetv2/lstm

best single model：4096x3 -> 1dcnn -> 512x256x3 -> resnet34
(private LB 0.8810, single fold with TTA)

# augmentation
- random shift wave separately, up to 1/32 second
- random change phase

# others
fine tune on cropped wave (ex. [1536:-256]) for more model blending
