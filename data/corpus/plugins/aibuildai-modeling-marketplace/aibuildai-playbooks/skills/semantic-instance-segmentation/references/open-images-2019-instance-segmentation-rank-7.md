# 7th place solution

Competition: open-images-2019-instance-segmentation
Rank: #7
Source: https://www.kaggle.com/c/open-images-2019-instance-segmentation/discussion/110983

Congratulations to all the winners. Here is our solution:

## Codebase and hardware

We used [mmdetection](https://github.com/open-mmlab/mmdetection) with 4 Tesla T4 during training, 8 Tesla T4 during TTA inference.

## Model setup

There are 300 classes, in 3 hierarchical levels: 
-  275 leaf classes
-  23 parent classes 
-  2 grandparent classes (`Carnivore` and `Reptile`)

For the 275 leaf classes, we train models with these 275 classes as labels and use inference results directly.

For the 25 parent and grandparent classes, there are two methods to get the predictions:
- (i). Use the prediction results from the leaf model, and “expand” to the parent and grandparent. For example, if the leaf model predicted a `Tortoise` mask, we add a `Turtle` prediction (its parent) and a `Reptile` prediction (its grandparent) with the same mask and score.  
- (ii). We train models with only the 23 parent classes as labels, and expand to the 2 grandparent classes. Note that for these models, we need to create hierarchical expansion of the instance segmentation before training (as explained [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md#instance-segmentation-track)).


## Validation set

We sampled a small subset from the official validation set for faster inference, 2844 images for the leaf model and 3841 images for the parent model. We used the validation scores to decide when to adjust learning rate, and to choose best checkpoints. 

The validation scores are significantly higher than LB. For the 275 leaf classes, we have validation score over 0.8. But nevertheless, the delta between validation and LB are stable.


## Training set rebalancing

The open images dataset is extremely imbalanced between the 300 classes. We rebalanced the training data for the leaf model as follows:
Sort the 275 classes by descending number of images. So class #0 is the largest class.

| Group | Class      | original num of imgs per class | rebalancing                   |
|-----|------------|--------------------------------|-------------------------------|
| 1     | #241 to #274 | 150-13                         | oversample x10                |
| 2     | #64 to #240  | 1500-150                       | oversample to 1500 imgs/class |
| 3     | #24 to #63   | 6k-1500                        | no rebalancing                |
| 4     | #0 to #23    | 89k-6k                         | downsample to 6k*            |

Each epoch has about 450k images after rebalancing. 

*For the downsample, we use different random seeds for each epoch to feed the model as many images as possible. Also, this 6k number is a rough target. In actual sampling, the top few classes ended up having more images because many images have multiple labels, and top classes already have more than 6k images after Group 1-3 sampling are done.

Similarly for the parent model, sort the 23 classes

| Group | Class    | rebalancing                  |
|-------|----------|------------------------------|
| 1     | #10 to #22 | upsample to 10k imgs/class   |
| 2     | #5 to #9   | no rebalancing               |
| 3     | #0 to #4   | downsample to 30k imgs/class |

Each epoch has about 330k images after rebalancing.

## Single models

We trained 3 cascade mrcnn leaf models
- L1: backbone x101
- L2: backbone r101 + deformable module
- L3: backbone x101 + deformable module

and 1 cascade mrcnn parent model
- P1: backbone x101

hyper-paramters:
```
imgs_per_gpu=1
num_gpu=4
img_scale=[(1333, 640), (1333, 960)]
multiscale_mode='range'
```

lr schedule:
model L1: 0.005 for 760k iterations; 0.005/3 for 60k iterations; 0.005/15 for 152k iterations
model L2: 0.005 for 675k iterations; 0.005/3 for 64k iterations; 0.005/15 for 48k iterations
model L3: 0.005 for 226k iterations; 0.005/5 for 132k iterations; 0.005/50 for 26k iterations
model P1: 0.005 for 148k iterations; 0.005/5 for 62k iterations; 0.005/50 for 8k iterations

Training took 0.5-0.65 hour per 1000 iterations with 4 T4. So in total the training the 4 models took about 22, 18, 10 and 5 days respectively.

Public/private scores of each (`max_per_img=120, thr=0`, without TTA):

| model | public score | private score |                |
|-------|--------------|---------------|----------------|
| L1    | 0.4708       | 0.4354        | on 275 classes |
| L2    | 0.4705       | 0.4275        | on 275 classes |
| L3    | 0.4712       | 0.4324        | on 275 classes |
| P1    | 0.0349       | 0.0344        | on 25 classes  |


## TTA

We used the TTA implementation from [Miras Amir's winning solution](https://github.com/amirassov/kaggle-imaterialist) of iMaterialist (Fashion) 2019 

For the leaf model, it’s the ensemble of 3 single models, 2 scale (1333,800) and (1600,960), and flip. At RPN and BB stages, it’s the NMS ensemble of 12 single models. In the end, it’s the mean of all 12 masks for each instance. 

TTA inference of the leaf model is very slow, which took about 525 T4-hours. We split the 99999 images into 25 chunks and ran it in parallel. 

For the parent model, it’s the ensemble of 2 scale and flip, but just one single model.

Lastly, for the 25 parent and grandparent classes, we implemented NMS ensemble at mask level (i.e. calculating the mask iou instead of bbox iou when determining which ones to suppress) to ensemble (i) leaf class predications expanded to 25 classes, and (ii) parent class predications expand to 25 classes.

Public/private scores of each (`max_per_img=120, thr=0`):

| model                    | public score | private score |                |
|--------------------------|--------------|---------------|----------------|
| leaf model ensemble      | 0.5007       | 0.4548        | on 275 classes |
| leaf ensemble expanded   | 0.0325       | 0.0330        | on 25 classes  |
| ensemble of (i) and (ii) | 0.0370       | 0.0370        | on 25 classes  |

At this point, the total score is public 0.5007+0.0370 = 0.5378, private 0.4548+0.0370 = 0.4918. 

There’s one last thing we did to boost scores to 0.5383/0.4922: set `max_per_img=200` for the leaf ensemble. But we only had time (also restricted by sub file size) to do this for 12/25 of the test images.


## Code

Training, inference, pre- and post-process code are available at https://github.com/boliu61/open-images-2019-instance-segmentation
Trained model weights are also linked in the readme there
