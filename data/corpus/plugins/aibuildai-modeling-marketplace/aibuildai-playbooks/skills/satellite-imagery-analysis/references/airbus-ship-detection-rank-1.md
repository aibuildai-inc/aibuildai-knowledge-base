# 1st place solution for Algorithm Speed Prize

Competition: airbus-ship-detection
Rank: #1
Source: https://www.kaggle.com/c/airbus-ship-detection/discussion/74443

Hi there!
Here's a quick breakdown of the 1st place solution:

0. PyTorch.
1. SE-ResNet50 as a classifier: took around 30 seconds to infer all images.
2. LinkNet as a segmentation network: took around 140 seconds to infer all positively classified images.
3. Extracting ship instances from binary mask via scipy.ndimage. Ignore instances with small area (less than 80px): took around 40 seconds.
4. Didn't use TTA.

Initial profiling has shown that I/O is the slowest part of the pipeline, so I've spent at least a week trying to optimize it. However, after a some rule clarifications, it was clear that I/O should be ignored completely, so I switched to optimizing the networks. After a bunch of experiments, I ended up with a slightly modified LinkNet, which was at least x10 faster than my model from the first stage of the competition.

#Key insights:

1. Adapt to the kernel. GPU kernel has 2 very slow CPU cores. I won 1 minute of inference time just by transferring image normalization to the GPU.
2. Classifier speed and accuracy is critical: it saves a lot of time, since segmentation network is much slower.
3. Predicting ship borders and/or using watershed requires too much post-processing.

#Training the classifier:

1. Train and predict on resized 224x224 images.
2. Nesterov SGD with LR 0.001, batch size 16, weight decay 1e-3, momentum 0.9.
3. BCE loss.
4. Augmentations: rotate90, flip, random brightness, gamma, bunch of blurs.
5. Use location-based stratification and split on 5 folds.
6. Balance the dataset (roughly 40/60 ratio between images with ships/without ships)

#Training the segmentation network:

1. Train in 3 stages: on 256x256 crops containing ships, then finetune on 384x384, and finally on 512x512. Inference on full-sized 768x768 images.
2. Augmentations: rotate90 and flip. LinkNet had a trouble converging with heavy augmentations.
3. Lovasz hinge loss with ELU + 1 trick.
4. Drop last residual connection.
5. Use location-based stratification and split on 5 folds.
6. Balance the dataset (roughly 40/60 ratio between images with ships/without ships)

#Some of the failed experiments:

1. FP16. K80 GPUs do not seem to support half-precision very well.
2. Separable convolution. Got marginal speed improvement on LinkNet, but the model couldn't reach good F2 score.
3. Simplifying larger network (U-Net based FPN with ResNet34). I used this model during the first stage of the competition. Couldn't get inference speed below 15 minutes.
4. Simpler classifier networks. I tried smaller ResNets, but couldn't get them to the same level of accuracy. What they gave in terms of speed, was then taken by false positives during segmentation.
5. Resizing images on GPU. Resizing itself was quicker on GPU, but transfer of large images from RAM and GPU was super-slow.
6. Replacing MaxPool layers with a strided Conv. Again, marginal speed improvement, but pretty low F2 score.
7. More aggressive pooling inside the network and upsampling at the final layer.

#If I had more time I would:

 1. Prune the network.
 2. Try to compete on CPU kernel, but with quantization and stuff. It would require using OpenVINO or PyTorch Glow.
 3. Reimplement scipy.ndimage on GPU. It took around 40 seconds of overall time.
