# 41st Place Solution: Cell-based RoI Pooling + Transformer Encoder

Competition: hpa-single-cell-image-classification
Rank: #41
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/238624

Hey fellow Kaggle competitors! 

I had a lot of fun with this unique competition and I'm looking forward to reading your own solutions to this problem. Here is an overview of my final approach + link to my code (open images in a new tab for higher res):

[Github Repo](https://github.com/martin-chobanyan/hpa-single-cell)

[cell transformer]

First off, in all of my models I used the pre-trained backbone CNN of @bestfitting 's winning DenseNet-121 models from the previous competition (3 different pre-trained versions). I stacked all of the color stains into 4-channel images and resized them to 1536x1536 and fed them through the backbone CNN, resulting in 1024 feature maps.

I then extracted a feature vector for each cell in the image by performing Region-of-Interest (RoI) pooling over the feature maps using the cell segmentation masks from the HPA Cell Segmentator. The pooling layer consisted of a concatenation of both avg-pool and max-pool features (resulting in a 2048-dim vector per cell). I also applied an adaptive average pool over the cell masks to reduce them to 8x8, which I then flattened. These downsampled, flatten masks served as position encoding for the cell regions (see diagram below):

[roi pool]

At this stage, we have a sequence of cell vectors, which allows us to explore the use of a Transformer model. The cell feature vectors are passed through a feed-forward layer which maps them to a 1024-dim embedding. Then, four encoder layers are applied as they were defined in the original Transformer paper. A final feed forward layer then maps each cell to an 18-dimensional logit.

To predict the image-level labels, a log-sum-exponential layer is applied which is defined as f(x) = (1/r) * ln(mean(e^(r*x))) with r=5 in our case. This serves as a mechanism to interpolate between average pooling and max pooling with different values for r.

For training I used Focal Loss with sigmoid activation and flip/rotate/crop augmentations. During inference, I extracted the cell-level logits and applied a sigmoid activation before their reduction to image-level labels. The final class prediction for a cell was averaged across all three models (three different pre-trained backbones).

I also explored the use of [Peak Response Maps](https://arxiv.org/abs/1804.00880) as a CAM based approach to the problem. Though the localizations looked good, the model was a bit too sensitive to classes which were not present in the image.

Things I would have tried with more time:

- Ensembling with a cell-level classifier
- Explore more ways of merging the peak-response-map results with the cell transformer model
- Domain adaptation to the public test set (since we know the test set has higher variability in protein locations within a given image compared to the training set).
