# Solution Sharing and Congratulations

Competition: the-nature-conservancy-fisheries-monitoring
Rank: #9
Source: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/discussion/31538#175317

Congrats to the winners and thanks sharing your solution and experience.
 
Mine uses Faster R-CNN with VGG for detection, a custom CNN for fish alignment and an ensemble of VGG16, ResNet50 and DenseNet161 for classification.
The detector is made up of a cascade of eight detectors, the subsequent ones taking no fish images of the former - the ones with better mAP being first.
The aligner was trained from scratch on 128px grayscale images. It predicts the fish head and tail, fairly well but many times interchanging them, and rotates the fish horizontaly. The aligned images are square sized
so they retain more background clutter then the bbox ones, still this helped quite a bit.
The classifier is made up an an ensemble of eight models, the mean of these gives the final prediction. All use pretrained models which I fine-tuned, mainly their affine, conv4 and conv5 layers.
I used Caffe and Python for all.
  
This was my first Kaggle competition and first ML project of any kind - my aim was to put in practice things learned in online courses like ML, NN, cs231n over the last year and to try out many different architectures.
Being my first project I spent quite a lot of time putting the whole pipeline together - from training data augmentation to emitting the submission file.
 
A lot of time I spent cleaning up the data due to mislabeling, learning how to differentiate these fishes, then accepting the fact that I myself cannot due to poor quality so many images had.
I also used about 20% of external images for training both the detectors and the classifiers, I think it helped.
I tried to balance the inequality in the number of images from different cameras with lot of augmentation - random rotations, zooming/shifting, salt-pepper noise, gamma changes. For this I prepared "recipes" of different proportions, trained and cross validated on these.
I think I should have spent more time on cross validation - used in the beginning but towards the end I selected my models mainly on how well their saliency map conveyed fishiness, striving for models that conveyed more details I tought to be relevant.
  
I had big expectations to Spatial Transformer Networks but proved mediocre at the end. I used STNs with 2 and 4 heads, similarly as they were used in the fine-grained classification of birds. I even tried to retain its localization head only and substitute the googLeNet based classifiers with VGG16, quite some monstrous thing came out that ate a lot of memory, but did not worked up to the expectations.
Still it was interesting to notice how well it learned to attend without any supervision to parts of fish relevant in classification, nice!

Unfortunately only in the last two days I tried DenseNets, these worked quite well, especially the 161 type, after only few hours of training gave a boost of 0.074 on the LB. They seemed to overfit slower on same datasets and their salicency map looked promising too.

Another thing that worked nicely was replacing pool5 with an SPP layer withing a VGG16.
Seems that more varied features out of the SPP and DenseNets helped quite a lot (what a surprise).

I missed out on trying the newer Inception v3 or v4 architectures, could not find pretrained models for Caffe and was not willing to train them from scratch and a promising thing called Neural Activation Constellations: Unsupervised Part Model Discovery with Convolutional Networks - this was in the last month and think I lost the zeal.

 All in all was a nice, challenging project to learn on, thanks The Nature Conservancy.
