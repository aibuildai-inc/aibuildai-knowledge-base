# 5th place solution

Competition: sp-society-camera-model-identification
Rank: #5
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49334

Our solution is pretty simple.

We trained several different CNN models and blended them. We were using imagenet weights for initialization and manipulations described on the Data page for augmentation.

**Key features:**

  - 224x224 random crops
  - TTA 11: all manipulations and rotates with geometric mean
  - Additional data (40k images from flickr which were filtered only by resolution)
  - Good predictions postprocessing (+3% accuracy)

One epoch takes only 8 minutes because of small crops. It allowed us to train several different models: Resnet50, Resnet101, Densenet121, Densenet201, Xception, Resnext101. Each one achieves 0.975-0.982 on private LB. Our final submission is just mode blend of these models :)

I uploaded all code to github repo: https://github.com/PavelOstyakov/camera_identification

There you can find an instruction how to train a model. I also uploaded weights for Resnet50. You can easily generate a submission which gives 0.98 on private LB.

Thanks to all participants and congratulations to the winners! It was a funny competition :)

Good luck in next competitions!
