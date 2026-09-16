# 8th Place Solution -- Decision tree part + source code

Competition: abstraction-and-reasoning-challenge
Rank: #8
Source: https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154436

Here's a description of the part of 8th place solution that @msypetkowski and I have developed.

Our solution is able to correctly predict:
* 8 / 102 on LB
* 76 / 419 on evaluation set
* 136 / 416 on training set.

### The source code: [GitHub](https://github.com/maciej-sypetkowski/kaggle-arc-solution), [Notebook](https://www.kaggle.com/msypetkowski/8-tasks-with-decision-trees-from-8-th-solution)

## Overview

In our solution we define each task as a pixel-level classification (10 classes) problem.
We featurize the input image several ways to handle different input and output shape relations.
We can distinguish 4 basic logic chunks:

1. Transformer -- transforms or augments a task. If needed, also transforms the prediction back after running next processing layers.
2. Featurizer -- converts a task (possibly transformed by a transformer) into a dataframe.
3. Model -- uses a dataframe returned by a featurizer. It is trained on samples that have output image, and makes predictions for test images. In the final solution we only use decision trees.
4. Confidence Model -- for every task we make predictions using all possible combinations of transformers, featurizers and models. After making multiple prediction, we have to select best 3 of them.

## Transformers

* Colorizers -- in many tasks, absolute colors of pixels don't matter, because the output image depends on pixels selected from the input image in some way. For example, consider a task which translates image one pixel to the left. It doesn't matter if the color of the pixel is red or blue, because we are supposed to assign the color of the input pixel that is on the right. In particular, tests images may contain colors that aren't contained in any of training images, and it can be not possible for standard classification models to generalize to it. Therefore, we re-colorize images for each sample in the task independently. The best performing colorizations are based on the number of occurrences of a given color, i.e. the most frequent color become the color 0, the second most frequent color become the color 1, etc; or the same with leaving the black color unchanged (because it's usually a background color).

* Color augmenter -- To extend the idea of colorizers, we perform a color augmentation. Colorizers sometimes can be not enough, for example when the test image has more colors than any of input images, because even after re-colorizing, the test image have colors that model has never seen. Color augmentation can help, because newly created samples can have all colors that are used in the test set. It can improve generalization of a model.

* Separation remover -- some tasks consist of _tiles_ and have lines separating these tiles. We use this transformer to remove these separation lines as they don't matter. In such tasks, the output size usually equals the size of a single tile, and this is handled by some featurizers.

* Flip &amp; 90 degrees rotation augmenter -- Some tasks are equivariant to flips or 90 degrees rotations (i.e. output of the flipped input should be the flipped output), and therefore we augment every input image in a task, producing many new samples, which can help with better generalization of the model.

Transformers can be stacked / combined with each other, and we try every combination of transformers taking one (or none) from every of the 4 groups.

## Featurizers

We designed a few different featurizers based mostly on different input-output size relationship. To do a featurization of an image, we have to first be able to predict the output size based on the input.

* `input_size = output_size` -- We perform a featurization of every input pixel and try to predict the output color. As features, in our final solution we use: self color; colors of 8 neighbors; X, Y, XY symmetry; colors of ray-cast hit in all 8 direction (we use 2 types of ray-cast -- one stops when encounters a different color, and the second when encounters a different than black color); double ray-casts (i.e. as previous but with additional second ray-cast in the same direction); lengths of those rays; the size of the component (i.e. the size of the one color figure to which the pixel belongs); the color surrounding the figure (only if the component is surrounded by exactly one color).

* `input_size` is a multiplication of `output_size` -- we have 2 featurizers for this case:
    * scale-like -- output pixels corresponds to contiguous fragments of the input image. Featurization of the output pixel is a concatenation of featurizations of every input pixel from the corresponding contiguous fragment.
    * tile-like -- the input image is divided into equally sized tiles, and `(x,y)-output_pixel` corresponds to the group of pixels consisting of `(x,y)-input_tile_pixel` from every tile. Similarly, featurizations are concatenated.

* `output_size` is a multiplication of `input_size` -- analogically to the former case we define two analogous featurizers. To the featurization, we additionally add coordinates of the output pixel in the group.
In this case one output pixel corresponds to one input pixel, but one input pixel is used for many different output pixels. Otherwise, output pixels corresponding to the same input pixel, wouldn't be distinguishable. Also we add symmetries depending on the parity of the coordinates in the corresponding group.

* `input_size^2 == output_size` -- if output size is squared input size, we can somewhat combine both scale-like and tile-like methods from the former case. For the output pixel we take concatenation of featurizations of input pixels on coordinates `(i,j)` and `(x,y)` where `(i,j)` are tile coordinates, and `(x,y)` are coordinates of the output pixel in this tile.

We experimented with extracting global features from images (e.g. based on object detection) and concatenate it to the rest. However, models tend to overfit heavily as those features are constant for every output pixel in one image.

## Models
We used mostly decision trees, as they can be easily interpreted and visualized. A decision tree can be considered a DSL which consists of checking conditions on features on the pixel-level.
In our final solution we use 2 trees -- one with gini and the second with entropy as split criterion.

We slightly modified `sklearn.tree.DecisionTree`. The default sklearn decision tree, in the case when multiple splits have the same score, it takes the random one. We modified this by assigning priorities to features, and instead of selecting the random one, we select the one with the highest priority -- if we get the same score when splitting by component size and self color, it's probably better to use self color.

We also tried using XGBoost, but it gave us the same score as decision trees. We sticked to trees as they require less memory and compute faster.

## Confidence model

After training models and making predictions from multiple different configurations, we have to select 3 best. At early stage of competition we sorted predictions based on the number of nodes in the tree -- if the tree is small and is able to unambiguously classify every training row, it should be good. Later, after adding augmentation, it was no longer effective (because the tree is larger when training with augmentations), so we decided to do a simple featurization of each configuration and train logistic regression on it. We call it confidence model, as for each configuration it predicts the confidence of being correct. As features we use: number of unique features in the tree; the number of nodes in the tree; whether or not each row in the train dataframe has one unambiguous output color; number of color permutation used for color augmentation; whether or not flip augmentation was used; whether or not flip + rotation augmentation was used. For the final solution, we trained a confidence model on the evaluation set.

## Additional ideas

We've been thinking about extending the idea of decision trees, but at the end we didn't have enough time to try it.
Vanilla decision trees return a concrete class as an output, which is not well suited for these kinds of problems. We think that it could be better if tree has one more kind of leaf, which tells "return the class that is in the i-th feature". Such tree may generalize better. It would be wise to consider a different split criterion, because output classes / leaf instances are no longer distinct, e.g. the leaf "return color 0" and the leaf "return self color" can sometimes give the same color depending on a sample, so gini impurity and information gain may not behave in a desired way.

In our final solution, the vanilla tree may struggle with learning an identity task (i.e. the output is exactly the same as the input). Our final model is able to learn such task because of color augmentation, but the tree would have to check every possible color and return the same. The above tree generalization would be just a stamp.
