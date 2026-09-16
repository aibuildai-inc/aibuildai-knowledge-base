# 1st Place Solution Write-up & Code

Competition: recursion-cellular-image-classification
Rank: #1
Source: https://www.kaggle.com/c/recursion-cellular-image-classification/discussion/110543

Thanks to Recursion and Kaggle for hosting such an interesting competition. It was really fun to participate.

**UPDATE: source code is here: [https://github.com/maciej-sypetkowski/kaggle-rcic-1st](https://github.com/maciej-sypetkowski/kaggle-rcic-1st)**

## Data pre-processing &amp; augmentation
* Loading original images (512x512)
* HUVEC-18 is moved to the training set (known leak)
* For training, all control images (also these from the test set) are used in the same way as non-control images
* Training augmentations
    * Random resized crop preserving aspect with scale ~ uniform(0.5, 1) using nearest-neighbor interpolation
    * Random horizontal and vertical flip, and 90 degrees rotation
    * Normalizing each image channel to N(0, 1)
    * For each channel: channel = channel * a + b, where a ~ N(1, 0.1), b ~ N(0, 0.1)
* Test-time augmentations
    * Horizontal and vertical flip, and 90 degrees rotation

## Model

* Backbone is pre-trained on ImageNet and first convolution is replaced with 6 input channel convolution
* Neck: BN + FC + ReLU + BN + FC + BN
* Head: FC

I found it important to not normalize input for the head, and that's the reason why the head and arc margin product are separate layers with different weights for fully connected layer (contrary to [what bestfitting did in Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109)).

## Training
* Batch size: 24 (48 with gradient accumulation)
* Optimizer: Adam
* Weight decay: 1e-5
* [Cutmix](https://arxiv.org/abs/1905.04899)
* Loss = ArcFaceLoss / 2 * 0.2 + SoftmaxCrossEntropyLoss * 0.8
    (ArcFaceLoss is divided by 2 to more or less preserve magnitude between losses)
* 90 epochs
* Learning rate: 1.5e-4 with cosine scheduling

## Post-processing
1. Predictions from different site images and different test-time augmentations are combined by taking mean of logits
2. Predictions for control classes are ignored
3. 831 classes that can't be on the given plate are marked as impossible
4. Linear Sum Assignment (LSA) is applied


With such configuration (training on all labeled part of the dataset -- no validation), I got 0.98997 private score and 0.95802 public score (single model).
Ensembling it with models trained in the same or very similar way (most of them with train/val split 5:1) (3x DenseNet161, 2x DenseNet161 with mixup (instead of cutmix), 5x DenseNet201 also with mixup, 3x ResNeXt50 also with mixup) gave me 0.99540 private and 0.98262 public (between 3rd-4th place on private LB).


To reach score of 0.997 private with single model I needed to add one more trick, which I would call:

## Progressive pseudo-labeling
In all write-ups I've read so far, pseudo-labeling methods consist of iteratively training new model(s) and enlarging training set using them. In my method, small amount of most confident predictions is pseudo-labeled and added to the training set **each epoch**. Precisely, for each epoch:
1. Predict all test and validation examples that weren't added to the training set yet (without TTA, only with combining over sites -- predicting with TTA could probably lead to a small improvement, but it would take more time to compute)
2. For each prediction, mark as impossible:
    * control classes,
    * classes that can't be on the plate,
    * classes that are already assigned to any image for the plate in the training set.
3. Select K most confident prediction (difference between greatest and second greatest class prediction)
4. Add new examples to the training set. If at least two examples are on the same plate and have the same class pseudo-labeled, add to the training set only the most confident one (to preserve uniqueness of classes on the plate)

During class assignment, I use greedy-like approach instead of LSA. However, to generate final predictions, the same post-processing as earlier (without pseudo-labeling) is applied (and LSA there). Using LSA increases score because examples that were added later are more difficult and have smaller confidence (because network saw them fewer times, and learning rate was smaller (decreasing learning rate policy)).

## Training previous single model further
* for additional 40 epochs,
* pseudo-labeling 40% of the test set at the start, and then adding 1.5% each epoch,
* cosine learning rate schedule with initial learning rate = 6e-5 scheduled for 60 epochs (i.e. at the end of training it is 1.5e-5),

gave me 0.99700 private and 0.99029 public, which already puts me on the 1st position.

Ensembling it with another model trained in the same way, but with train/val split 5:1, I got 0.99749 private and 0.99187 public. Adding one more model to the ensemble (also trained in the same way, but on the different split) gave me 0.99763 private and 0.99187 public, which matches my private score.

I noticed that around 120th epoch (30th epoch of pseudo-labeling -- 85% of test set already added) first pseudo-label misclassifications on validation set started to occur, hence I tried to fine-tune model even further (taking checkpoint from 120th epoch), starting with 80% of test set, and again incrementally adding new images for 30 epochs. And then again taking checkpoint 5 epochs before end, and starting from 95% for 20 epochs. However, by doing this I was able to classify correctly only one private example more (0.99707) and increase public score (i.e. probably only U2OS-04 experiment) to 0.99232.

My final submission is an ensemble of 11 models (6x DenseNet161, 5x DenseNet201) (each of them with pseudo-labeling) with more TTA (also predicting on crop-resized images with scale 0.75 and 0.85), but it didn't give me any boost on the private test set (0.99763), but helped on the public test set (probably only on U2OS-04) (0.99480).



## Other insights
* Mixup performs a little better than cutmix on the part without pseudo-labeling, but it converges slower. On the contrary, with pseudo-labeling, cutmix was a little better (probably because of faster convergence)
* Larger architecture is better: DenseNet121 &lt; DenseNet169 &lt; DenseNet201 &lt; DenseNet161 -- for people without knowledge about DenseNets: DenseNet161 has less layers but more parameters than DenseNet201
* EfficientNets and ResNeXts didn't work for me


## Attempt to use control images in a smarter way
I want to share the approach I've tried, however it didn't give me any boost in the score, and I didn't use it in the final submission. But I think it's very valuable information, especially for the further research.
My idea was to instead of feeding to the head embedding only, feed also some information about any reference image from the same plate/experiment (e.g. its embedding and one-hot label).
Using only control images as a reference, would lead to overfitting. To tackle that problem, I used also non-control images as reference -- during training, control and non-control images are treated in the same way; during inference, only control images are used as reference (obviously, non-control images in the test set are not labeled).
We have 1139 classes per experiment (or 277 + 31 = 308 per plate), so the head would see each pair of classes after 1139 * 1139 = 1297321 images (once per 15 epochs) or 308 * 308 = 94864 (once per 1 epoch). To solve this, I ensure that in every batch there will be constant number of images from each of randomly chosen experiments/plates. For example, for batch size = 48, I can have 8(number of experiments/plates) x 6(number of images from given experiment/plate), and run the head on each pair among each experiment/plate in the batch. That gives 8 * 6 * 5 = 240 pairs in one batch (I forbid the image and the reference to be the same image), and doesn't increase training time (embedding of each image is calculated only once, and the head consists of few fully connected layers).

However, it didn't work any better than normal classification, and sometimes even worse. I tried to add or modify features for the head, for example:
* concatenate difference between / multiplication of image and reference embedding,
* concatenate corresponding vector from the arc margin product layer to the reference label,
* normalize / not normalize embeddings,
* detaching some of the features (not computing gradient through them).

I also tried:
* add more layers to the head,
* heavy-augment all images in a batch for the same experiment/plate in the same way -- all images in the batch belonging to the same experiment/plate would have the same brightness, contrast, gamma correction, the same scale applied, and so on -- the idea was to artificially simulate other cell types to direct model toward using references more effectively,
* use mixup only within images from the same experiment/plate.

However, no luck.

What more, after training such model and feeding random noise as the reference, network still inferred very similar predictions with almost the same validation score (and not always worse). So, network didn't learn how to use the references properly.
That would imply that creating a model that performs well on different cell types (not seen during the training) using control images may be a very hard and challenging problem.
