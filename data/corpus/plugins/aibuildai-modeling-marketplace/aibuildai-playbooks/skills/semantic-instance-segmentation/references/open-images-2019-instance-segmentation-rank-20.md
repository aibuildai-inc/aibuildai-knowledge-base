# 20th place solution - maskrcnn-benchmark baseline

Competition: open-images-2019-instance-segmentation
Rank: #20
Source: https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/110908

Thanks to Open Images and Kaggle team for this great competition(s) and congrats to all (tentative) prize and medal winners!

My result is not outstanding but the solution might be valuable to be shared because I used the famous maskrcnn-benchmark library 'as it is' and also used its outputs as it is without TTA or any post processing. Training two models requires only 14 hours (x2) using V100 8GPUs.

All codes are available at: https://github.com/yu4u/kaggle-open-images-2019-instance-segmentation

There are mainly two issues to be solved in this competition and the Object Detection track: (1) class imbalance and (2) class hierarchy. I tackled these issues only on a dataset creation side. The former is easy to handle: use fixed number of training images for each class. In this post, I mainly describe how to handle class hierarchy.

Firstly, I divided all classes into two groups: layer0 and layer1. From challenge-2019-label300-segmentable-hierarchy.json we can see that:
1. Maximum depth of hierarchy is 2 (starting from 0)
2. The number of depth 2 classes is only 5.

```
Carnivore
└── Bear
    ├── Brown bear
    ├── Polar bear
    └── Teddy bear &lt;---　Are you serious?

Reptile
└── Turtle
    ├── Tortoise
    └── Sea turtle
```

Thus, I decided to group depth 0 classes as layer0 group and depth 1 and 2 classes together as layer1 group. The idea is to make different model for each of two groups.
In training each model, a dedicated dataset is used, which includes only the target group class instances. By doing so, there is no need to care about class hierarchy.
However, practically, it is impossible to make dataset from only training images that includes only target classes and does not include non-target classes. Therefore, I removed non-target class instances from training images.

For layer0 group dataset:

1. Remove non-target class annotations that occlude target class object 25% or more
2. Convert non-target class to its parent class (Thus it becomes target class. Some classes need to be processed twice. 'Teddy bear' is converted only to 'Toy', not 'Carnivore')

For layer1 group dataset:

1. Remove non-target class annotations that occlude target class object 25% or more
2. Remove non-target class annotations that do not have any child class (no impact to layer1 group classes because there is no relationship between them)
3. Remove non-target class annotations that have some child classes, and fill their bbox with gray in the training image (removing only annotations is not good idea because these cause 'false false positive' signal (loss) to the model)

That's all, and let's train!
