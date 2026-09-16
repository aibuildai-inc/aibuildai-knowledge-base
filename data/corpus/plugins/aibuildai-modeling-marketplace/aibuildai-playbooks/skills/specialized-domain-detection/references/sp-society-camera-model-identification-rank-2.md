# 2nd place solution with GPU muscles

Competition: sp-society-camera-model-identification
Rank: #2
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49299

**tldr**: 
We downloaded a about 500+ GB photos, trained 9+ imagenet-like models with 3 version of the pipeline. Finally, we averaged 27 checkpoints with geometric mean.

**Key components of a good solution, by priorities:**

 1. Large and clean external dataset 
 2. Consistant local validation
 3. Classic competitive approach to learning models
 4. Diverse models

**Data mining**:  We downloaded 500+ Gb photos from various resources: Flickr, Yandex.Fotki, Wikipedia Commons, mobile reviews. In addition, on the last night we downloaded 22k more photos with urls from Flickr gathered by [Andres Torrubia][1]. 

**Filtering data**: Lightroom/photoshop/etc processing could eliminate all the information about the camera. We filtered on: model, resolution, quality of jpeg compression, software of processing.
After filtering of the training and validation, the datasets looked like this:
![enter image description here][2]
We took the validation from [Gleb’s post][3], but replaced iPhone 6 plus pictures in it by iPhone 6 ones.

**Training models**: Our code is based on pytorch version of [Andres solution][4].
For all the models, the binary flag is_manip was used as an additional feature for the classifier of a net. All models had an input of 480. For the majority of our models, five crops + flip photo orientation (10TTA) was applied to the pictures, for D4 models five crops + the whole group of D4 were applied (40TTA); then geometric mean was applied to the predictions.
Useful tricks:
1. Adam, reducing LR on plateau with patience 2-4
2. Cyclic LR with SGD
3. Pseudo-labeling
4. Averaging 3 checkpoints with the best loss for validation

Also on the last day we trained several models with D4 augmentations, finetune from best checkpoint of previous pipeline. We didn’t have submit to check all models on LB, but the result on one was impressive.

The final ensemble of models looked like this:
![enter image description here][5]
Logging models:
 ![enter image description here][6]
**Averaging**: We tried different approaches with class balancing and dropping the missing classes according to the probabilities. But in the end geometric average of 27 checkpoints of different models is the best.

**What did not work?**
Demosizing
![enter image description here][7]
Cameras fix on the matrix in a particular pixel the intensity of only one color, and the rest of the colors are restored to neighboring pixels, and the recovery algorithm for different camera manufacturers is different.
The idea was to calculate the difference between the original image and the image obtained by one of the standard demosayzingov.
The inertia of the differences is to train the neural network.

**Hardware:**

 - i7 7700k, 64gb, 2x 1080 (just for development) 
 - i7 6700k, 32gb, 2x    Titan X (Maxwell)
 - i7 5930K, 32GB, 3x1080Ti
 - Xeon 2696v3, 64gb, 4x1080Ti
 - i7 3770k, 16gb, 2x 1080Ti

  [1]: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49064
  [2]: https://pp.userapi.com/c824701/v824701072/a95a8/Hgl68WyLzyU.jpg
  [3]: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/47235
  [4]: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/48679
  [5]: https://pp.userapi.com/c824701/v824701072/a95b9/dg36oZu6_B8.jpg
  [6]: https://pp.userapi.com/c824701/v824701072/a95b2/aMQ2rKhWAI4.jpg
  [7]: https://pp.userapi.com/c824701/v824701072/a95c0/sJpJNsYtDcg.jpg
