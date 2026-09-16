# 11st Place Solution Summary: Active learning for cell label mining and Ensemble of different models

Competition: hpa-single-cell-image-classification
Rank: #11
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/239211

First of all, I would like to thank Kaggle and organizers hosting this really interesting competition. Thanks to my teammates @silversh and @jianguoguo  for the hard work and collaboration. 

Congratulations to all the winners. This is my first Kaggle competition, I really learn a lot of nice ideas and solutions in the forum.

**1. Overview**
Our final solution includes active learning for cell label mining and ensemble of different models.

**2. Active learning for cell label mining**
Because I do not participate Kaggle conpetition before, I'm not sure whether data label mining is allowed. Therefore, we try to use active learning to mine cell labels relatively late. Due to time limitation, we only apply it for a few classes, i.e., 9, 11, 12 and 15.

First, we chose the cells in the single-label image as the initial trainset, and the cell labels directly inherit the image labels. For classes 11 and 15 with few images, we manually selected about 200 cells with ground truth labels into the initial trainset. After the model is trained on the initial trainset, it is used to make predictions for cells in all images. Use the following strategies to prepare the trainset for the next round.

If an image contains original label X and the prediction confidence of one cell in this image is greater than 0.25, the label X of the cell is retained, otherwise the cell is removed from the next round trainset. If an image does not contain original label X, and the predictive confidence of one cell in this image is greater than 0.5, this cell will be assigned a true label X. 

After several rounds, the noise of most cell labels can be inhibited. For example, we finally retrieve about 2,000 cells for class 11 after 4 rounds. On the final public test set, the AP score for class 11 is improved from 0.001 to 0.035.

The specific principle of active learning can be referred to my previous papers [https://doi.org/10.1016/j.future.2019.07.013](url).

**3. Ensemble of different models**
We use ensemble method to fuse Cell-level models and Image-level models. The weights are 0.75 and 0.25, respectively. The Cell-level models employ EfficientNet-B3 network, and the cell score is averaged after 5fold cross-validation. For the Image-level models, we train EfficientNet-B7 networks with RGBY 4-channel image input and Green 1-channel image input, respectively, and the image score is averaged from the two Image-level models. Specifically, when the Cell-level models and the Image-level models are fused, the prediction result of class 11 only use the prediction score of the Cell-level model.

**4. Configuration of training**
- Image-level model:
1.  EfficientNet-B7，Concat-pooling + 2 layers of FC Head, input size 600 × 600
2. Data augmentation: flip, rotation
3. Focal Loss
4. SWA (Nearly useless)
5. Fusion of RGBY 4-channel model and Green 1-channel model

- Cell-level model:
1.  EfficientNet-B3，Concat-pooling + 2 layers of FC Head, input size 300 × 300
2. Data augmentation: flip, rotation, mixup
3. Focal Loss, Label smooth [0.1, 0.9]
4. SWA (Nearly useless)
5. 5-folds cross validation

**5. Some results**
| One Cell-level model only | active learning for classes | Public LB | Private LB |
| One Cell-level model only | 11, 15 at 1st round               | 0.446       | 0.465        |
| One Cell-level model only | 11, 15 at 2nd round             | 0.462        | 0.494       |
| One Cell-level model only | 11, 15 at 4th round              | 0.502        | -               |

| Ensemble model        | active learning for classes | Public LB | Private LB |
| Ensemble model        | 11, 15               | 0.558       | 0.528        |
| Ensemble model        | 9, 11, 12, 15     | 0.558       | 0.532        | -> Final result in LB
| Ensemble model        | 9, 11, 12, 15 at 4th round, 4 at 1st round | 0.561        | 0.537       | ->Notebook timeout


**6. Others**
TTA is helpful for prediction, with an increase of 0.001~0.003, but it may tend to time out for our submissions.

Early in the competition, we tried to solve this problem by using multi-instance learning with transformer encoder-based attention, but the effect was not good.

As the results above, we additionally use a round of active learning for cell label mining on class 4. The submission was timeout and we resubmited it after the deadline of the competition, and it finally obtained the results of Public LB of 0.561 and Private LB of 0.537, indicating that active learning mining labels can continue to improve the performance if it was extended to other classes. However, we adopted this strategy too late..

Please forgive my typesetting, I encounter problems inserting images and tables...
