# 9th Place Solution

Competition: sp-society-camera-model-identification
Rank: #9
Source: https://www.kaggle.com/c/sp-society-camera-model-identification/discussion/49319

First of all, congratulations to all of my teammates: [ternaus][1] for getting Grand Master title; [arsenyinfo][2] and [mephistopheies][3] for Master titles and [cortwave][4] for the first gold medal. Well-deserved guys!

Here is a brief overview of our solution:

 1. Each of us created a solid single model earlier in the competition with the initial data and Gleb's data. There were MobileNets, VGGs, custom ResNets, DenseNets (about 6-7 models) with different crop sizes, augmentations and TTAs. We constructed majority voting out of these models and put test images with the maximum votes to pseudolabels.
 2. Got more data from flickr: about 40k images. They have been filtered by camera model, resolution, quality and any software changes. Constructed validation from Gleb's and new flickr images.
 3. The major idea of our models was to make training process iterative. In the first stage we've used train+flickr data (Gleb's and ours). Adding pseudolabels and resetting LR in the second stage. In particular, my own approach included: in the first stage, training VGG-16 on initial images + some flickr (overall 7k images). In stage 2, keep only pseudolabels and highly overfit to them reaching 100% accuracy on the train set. Such model gave 0.985 Private LB.
 4. Occasionally, we noticed that distribution of test photos is quite uniform in the test set and decided to force it to be exactly uniform. [mephistopheies][5] made some Analysis magic and gave a formula for such a normalization (it's better to ask him directly what he's done :) )
 5. Our final submission was a blend of 10 models (with and without stages) and subsequent classes balancing.

Repo of our solution:
https://github.com/cortwave/camera-model-identification

TL; DR:

Did Work:

 - Pseudolabels
 - External data and data cleaning
 - Different crop sizes 
 - Picking the argmax of probabilities during TTA
 - Balancing helped in the Public LB, but now we see that is has been overfitting

Did Not Work:

 - Averaging model checkpoints
 - Training on denoised images
 - Non-standard loss functions like hinge loss
 - KNN on pseudolabel embeddings
 - GAN and Siamese architectures

P.S. Late submission of the blend on top-3 models scored 0.989 Private LB. Unfortunately, we haven't tried this one due to the lack of submissions..


  [1]: https://www.kaggle.com/iglovikov
  [2]: https://www.kaggle.com/arsenyinfo
  [3]: https://www.kaggle.com/nesterov
  [4]: https://www.kaggle.com/cortwave
  [5]: https://www.kaggle.com/nesterov
