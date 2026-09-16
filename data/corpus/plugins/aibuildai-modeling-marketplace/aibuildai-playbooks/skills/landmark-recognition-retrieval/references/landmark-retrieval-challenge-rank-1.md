# 1st Place Solution Summary: CVSSP & Visual Atoms (0.627)

Competition: landmark-retrieval-challenge
Rank: #1
Source: https://www.kaggle.com/c/landmark-retrieval-challenge/discussion/57855

Hi everyone,

Congratulations to everyone else who participated for making it such a fierce competition :)

Here’s a detailed summary of our team’s solution to the competition for those of you who are curious. Our solution consisted of two main components: First, creating a high performance global descriptor that can represent the images in the dataset as singular vectors, and then building an efficient scheme for matching these vectors to find the most likely matches to submit to the leaderboard.  
Below is a high level flowchart overview, with LB scores labelled at each applicable step, followed by a detailed description of each component of our solution.

![Complete solution flow diagram][1]

NB: Throughout this post we refer to scores as percentages for clarity, i.e. 62.5% = 0.625. All scores are on the public LB.

----------
# In-depth

## Global Descriptors
The main part of our solution involves several global descriptors - vectors which describe the entire contents of the image. We start off with two pre-trained CNN models ([ResNet and ResNeXt](https://github.com/facebookresearch/ResNeXt)), and use four state-of-the-art aggregation methods to generate global descriptors from these models:
Here are brief details of each method and its “raw” performance (i.e. without query expansion, database expansion, etc):

 - **Region-Entropy based Multi-layer Abstraction Pooling (REMAP) [42.8% mAP]:** Our latest design for a global descriptor, which aggregates a hierarchy of deep features from different CNN layers, trained to represent multiple and complementary levels of visual abstraction. We plan to present the details of our REMAP architecture at the forthcoming CVPR workshop.
 - **Maximum Activations of Convolutions ([MAC](https://arxiv.org/abs/1511.05879)) [32.9% mAP]:** MAC descriptors encode the maximum local response of each of the final layer convolutional filters. In the MAC architecture, the last convolutional layer of ResNeXt is followed by a max-pooling layer, L2-normalization layer and PCA+Whitening layer.
 - **Sum-pooling of Convolutions ([SPoC](https://arxiv.org/abs/1510.07493)) [31.7% mAP]:** In the SPoC pipeline, the last convolutional layer of ResNeXt is followed by Sum pooling layer, L2-normalization layer and a PCA+Whitening layer.
 - **Regional Maximum Activations of Convolutions ([RMAC](https://arxiv.org/abs/1610.07940)) [34.7% mAP]:** In RMAC, the last convolutional features of ResNeXt are max-pooled across several multi-scale overlapping regions. The region based descriptors are L2-normalized, PCA+Whitened and L2-normalized again. Finally, the descriptors are sum-aggregated into a single descriptor.

The underlying CNN networks (ResNet and ResNeXt) are pretrained on ImageNet, and fine-tuned on a subset of the [Landmarks dataset used in the work of Babenko et al.](http://sites.skoltech.ru/compvision/projects/neuralcodes/), that contains approximately 120k images of 650 famous landmarks sites.

The images in this dataset were originally collected through textual queries in an image search engine without thorough verification, and therefore they contain a significant number of unrelated images which we filter out and remove. This procedure is semi-automatic, and relies on dense SIFT features detected with a Hessian-affine detector and aggregated with the [RVD-W descriptor](http://epubs.surrey.ac.uk/812468/). This cleaning process leaves about 25,000 images still belonging to one of the 650 landmarks, which is what we then used to fine-tune our models for this competition.

We didn’t use the data from the landmark classification sister competition for training, as we wanted to see how well our solution would generalise to new dataset (as opposed to creating a solution which was specifically tailored to this dataset).

## Combined Descriptor
Our final global descriptor was built by concatenating six fine-tuned global descriptors trained as explained above (LB score shown in brackets):

 - ResNeXt+REMAP (**42.8%**)
 - ResNeXt+RMAC  (34.7%)
 - ResNeXt+MAC  (32.9%)
 - ResNeXt+SPoC  (31.7%)
 - ResNet+REMAP  (35.8%)
 - ResNet+MAC   (30.4%)

We assign each descriptor a weight by scaling them to a fixed L2 norm, and concatenate as follows:

    XG = [2× ResNeXt+REMAP;  1.5× ResNeXt+RMAC; 1.5× ResNeXt+MAC; 1.5× ResNeXt+SPoC; ResNet+MAC; ResNet+REMAP]

The weights are selected ad-hoc to reflect the relative performance of each method. After this, we perform PCA to reduce the dimensionality of the descriptor to 4K (not just to save computation, but also to discard noisy dimensions) and apply whitening so that all dimensions have equal variance.
While PCA and whitening only gave a small improvement on its own, it improved the result of our query expansion step by several %.

## Nearest Neighbour Search
After descriptor creation, each image is represented by a 4096-dimensional descriptor. Next, we use an exhaustive k-nearest neighbour search to find the top 2500 neighbours and L2 distances for each image – nothing fancy here. Submitting the top 100 neighbours for each test image at this stage nets a score of **47.2%**.

This step is implemented using optimised NumPy code and takes 2 hours to find the top 2.5K neighbours for each of the 1.2M images (we needed the nearest neighbours for the index set too for the next steps).

## Database augmentation
The next step is to perform [database-side augmention](https://arxiv.org/abs/1610.07940) (DBA), which replaces every image descriptor in the database (including queries) with a weighted combination of itself and its top 10 neighbours. The objective is to improve the quality of the image representations by leveraging the features of their neighbors. More precisely, we perform weighted sum-aggregation of the descriptors, with weights computed using:

    weights = logspace(0, -1.5, 10)

Interestingly, while on other datasets we find that using more than a couple neighbours for augmenting the _query_ is detrimental to the score, we found 10 neighbours to be the optimum for both query and database images for this dataset.

It should be noted that DBA was the last thing we added to the pipeline, and while it gave a huge improvement to our score on its own, when combined with the query expansion step, the improvement is only a modest 1-2%. We believe this is because the DB expansion acts very similar to the first level of our query expansion method.

## Query Expansion
Query expansion is a fundamental technique in image retrieval problems which often nets a significant improvement in performance. It works off the principle that if images A &lt;-&gt; B, and B &lt;-&gt; C, then A &lt;-&gt; C (even if descriptors A and C are not explicitly matched). A simple example where the benefit of this is illustrated is in cases where there are three partially overlapping images:

![Illustration showing query expansion][2]

In this case, a query expansion scheme can help us match images A and C as being of the same scene, even though the descriptors themselves (especially global descriptors) are very unlikely to match. This also benefits us in other cases where, for example, we have several images with gradually different perspectives or lighting conditions, and intermediate images can help us to connect them.

In this competition, we designed a novel, fast technique for query expansion which can be run recursively to capture long-distance connections between images with many intermediate steps, which lends itself well to this problem as some landmarks have very many images - only one image has to be matched to a query and all the images of the same landmark are then also appended to the top 100 list.

We’ve opted not to discuss the exact method here and instead explain it in-depth in a paper in the coming weeks. But the basic principle is the same as other well-established query expansion techniques. Our query expansion gave an approx **11%** improvement when run for one iteration, and **14%** improvement when run recursively (with 30min runtime) - this is reduced when combined with DBA.

## Simplified Model

In a production ML environment, it’s rare to use a monster ensemble of various models when the laws of diminishing returns apply - Usually most of the performance can be achieved with a small subset of the complexity. We expect a slimmed-down version of our solution with just ResNeXt-REMAP and query expansion to score 56-57% with &lt;12 hours total runtime from downloading the data to submission (on 4 GPUs) - largely aided by the fact that it did not have to be trained for this competition.

![Simplified model flow diagram][3]

## Things that didn’t work

 - **Local descriptors:** This was perhaps the most surprising thing for us in the competition. We tried several schemes based on various local descriptors, with or without geometric verification, for example using it for reranking our results (decrease in performance), or using it to scan through the top few thousand global neighbours to find very confident local matches that the CNN-based globals missed (helped, but the improvement was sub-0.1%). We’re very curious if/how other teams were able to harness the power of local descriptors in this competition. Perhaps CNN-based global descriptors are so good now that the era of locals is over?
 - **Coping with rotated images:** A glance through the dataset showed us that a significant number of images were rotated. We tried accounting for this in multiple ways, for example by also comparing rotated and non-rotated descriptors during kNN and taking the closest match for each pair of images, however we could not increase our score. This is probably because as this is a dataset with very many distractors, the increase in false positives outweighs the gain in true positives.
 - **Ensembles:** We tried various methods to combine very different submissions with methods such as rank averaging and interleaving predictions, however the gains we saw from this were very small. It seemed to work better to combine models earlier on within the solution than at the end.

## Acknowledgments

Components of our solution have been supported by work done under the industry project "iTravel - A Virtual Journey Assistant",  InnovateUK grant #102811 (funder: UK Research and Innovation).  
As this is an industry-led project, we are unfortunately unable to release the code for our solution under the Apache 2.0 License (which includes royalty-free commercial use).


----------


Finally, I would like to say thank you to Kaggle and the organizers for organising this very interesting competition - it is unusual and refreshing to see both retrieval and semi-supervised learning competitions on Kaggle. I would also like to thank my excellent teammates Sameed and Mirek for their hard work and fun moments and everyone at both [CVSSP][4] and [Visual Atoms](https://www.surrey.ac.uk/centre-vision-speech-signal-processing) for tracking our progress and supporting throughout. The whole competition has been a very enjoyable experience for us.

We will do our best to answer any questions you might have about our solution, and we look forward to seeing the solutions of the other teams :)

\- Mikel (with the team)  
CVSSP &amp; Visual Atoms

  [1]: https://i.imgur.com/GksVSIB.png
  [2]: https://i.imgur.com/wMCa17s.jpg
  [3]: https://i.imgur.com/zWcCl2a.png
  [4]: https://www.surrey.ac.uk/centre-vision-speech-signal-processing
