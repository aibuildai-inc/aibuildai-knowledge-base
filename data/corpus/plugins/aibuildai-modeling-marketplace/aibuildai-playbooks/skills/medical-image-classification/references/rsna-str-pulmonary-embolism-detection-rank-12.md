# 12th Place Solution

Competition: rsna-str-pulmonary-embolism-detection
Rank: #12
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193415

First, we would like to thank all the organizers, Kaggle and all the medical institutions who contributed their data and all the data annotators. As someone who previously tried to annotate an MRI scan during my internship I know a bit about how cumbersome and sensitive it's to annotate medical data. So appreciate it all!

In overall it was a great yet challenging competition in terms of the data volume. We initially started training models from data shared by @vaillant, without his generosity the barrier of entry to this competition would be too high for many participators. So, a special thank to him goes from our team.

Although, starting off with 256x256 images was great for prototyping in computer vision problems you can always get significant boosts just by training with higher resolution images. At first we tried to create and save full resolution images using Kaggle kernels but it wasn't fun and easy since only 5GB disk space is allowed so we ended up using a cloud provider for the remaining experiments.

First, we created full resolution training images using the same windowing shared publicly and also leveraged great utilities from https://docs.fast.ai/medical.imaging. GDCM was also a requirement, because not all images were readable without it.

We extracted both images and metadata.



Later we trained CNN models for predicting Image level PE.



Then we used an LSTM model to predict image level PE and exam level predictions.




[Inference Kernel](https://www.kaggle.com/keremt/12th-place-rsna-pe-inference?scriptVersionId=45505224)

Other Notes:

- 5 folds validation scheme.
- Sequence model directly optimized on competition metric.
- Tried EfficientNet but we had problems with overfitting.
- Didn't have time for stacking experiments.


Code for this competition will be publicly available in this [repo](https://github.com/KeremTurgutlu/rsna-pulmonary-embolism). 

Special thanks to my teammates: @jesucristo, @josealways123 and @atikahamed
