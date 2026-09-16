# 2nd place solution

Competition: understanding_cloud_organization
Rank: #2
Source: https://www.kaggle.com/c/understanding_cloud_organization/discussion/118255

Hello to everyone participating in the competition, congratulations to all who won and thanks to kaggle for the excellent competition.

Here I will give a general solution to the problem, I will talk about techniques that helped and those ideas that did not work.

Most recently, I participated in kaggle segmentation contests 
[SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) and [Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection)
Therefore, I have gained decent experience in solving such problems. I already had an idea of what could work and what couldn’t.
** **

### Idea #1
Looking at the data, I saw that the images have a dimension of 1400x2100 and it was not a good idea to put such data to the network directly. Of course, it was possible to resize the image to 2 or 4 times, but obviously, we will definitely lose something from the data.
I came up with a compromise. Use a small network - a compressor, that extracts significant features from the data and reduces the image size.
It looks something like this:



To build models, i used Keras 2, Tensorflow 1.4 and the library [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models) (thank you very much Pavel Yakubovskiy)
** **

### Idea #2
In order to build an effective ensemble, we must use models with the least possible correlation between predictions. I decided to use such combinations of model parameters:



All models had a Unet decoder.

** **

**training parameters:**
Optimizer: Adam
Loss Function: FocalLoss
Batch Size: 4

Hard albumentation: 
Hflip, VFlip, Equalize, CLAHE, RandomBrightnessContrast, RandomGamma, Cutout
ShiftScaleRotate, GridDistortion, GaussNoise

30 epochs on a two-cycle learning profile. It looks something like this:



For training models, I used 2xP3.2 Amazon instance
** **

### Idea #3
**Postprocessing**. 
Mean average all models -&gt; raw probability
All tasks for segmenting objects with a DICE metric are very sensitive to FalsePositive errors. In some cases, training a separate classifier model for detect of a mask in the image very helps. In my case, the classifiers did not help much and I used the Triple rule method, which I first saw in the first place solution about competition [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation). 
Thanks so much for the idea of **Aimoldin Anuar**  https://www.kaggle.com/sneddy
The description of this approach can be understood from here [Kaggle SIIM-ACR Pneumothorax Challenge - 1st place solution - Anuar Aimoldin](https://youtu.be/Wuf0wE3Mrxg)


The triple rule parameters (threshold1, minsize, threshold2) were searched by global optimization methods.

Basically, this is all that helped in solving the task.

What didn't work:
- Mask classifiers
- mmdetection / FasterRCNN
- BCE-DICE, lovasz, triple_loss
- Adversarial validation
- Pseudo labeling


&gt; 
Thanks for watching
