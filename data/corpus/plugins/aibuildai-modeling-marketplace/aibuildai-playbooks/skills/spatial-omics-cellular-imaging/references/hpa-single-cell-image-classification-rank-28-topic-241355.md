# 28th solution/ with self-supervised learning

Competition: hpa-single-cell-image-classification
Rank: #28
Source: https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/241355

Thanks to the organizers for this very interesting competition, and we have learned a lot during this competition.

Code link: https://github.com/tuotuo-1997/HPA

**Method Overview**

We use the Kaggle provided dataset and the public dataset to train and evaluate using different model architectures. The public tools used include Fastai, Opencv, CellSegmentator, Cleanlab, etc.

We classify 12 models from the whole image and cells. The 8 different model structures used in the cell classification are: Resnet18, 34, 50, 101, Densenet121, Efficientnet B0, B1, and B2. The cell label assignment is consistent with the whole image. The four different model structures used in the whole image are: Efficientnet B0, B1, B2, and B3.
We use Adam as the optimizer. For the cell model, each model trains a total of 4 epochs, and for the whole image model, each model trains 6 epochs. The initial learning rate is set to 3e-2. The predicted probabilities of the cells and the whole image are averaged to obtain the final result.

The improvement of results in our method is mainly concentrated in the following 5 points:
1) Self-supervised learning (SSL).
2) The HPA Public data set.
3) The combine of cells level and whole image level results.
4) Asymmetric Loss.
5) Modify the label of the cell to 18 whose maximum value of the green channel is less than 60.
6) Ensemble.
The training time of the entire 12 models on 8*1080Ti is about 48h. It takes about 1.5 hours with a single 2080Ti to submit  to Kaggle (for public test dataset). After the code is submitted, the prediction time of Kaggle kernel is about 9 hours (for all test datasets).

**Conclusions**

It is important to use Asymmetric Loss can alleviate category imbalance while SSL can be helpful to improve the performance. The combine of the cells and the whole image could also greatly improve experiment result.
