# 12th place solution

Competition: human-protein-atlas-image-classification
Rank: #12
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77325

Hey kagglers, you know what they say: Be careful of overfitting, rare classes and bestfitting! Hah.
 
Jokes apart, congratulations to all the winners, this was a really tough competition. Thanks to the Kaggle organization and Human Protein Atlas for providing such a great challenge and also thanks to @brian @heng and @tilii for all the useful posts, I have learnt a lot from you. 

I am going to briefly detail my solution:


##Hardware##
  First I had a 980Ti GPU, 2600k CPU and 14GB RAM. Four weeks ago I bought a 1080Ti.

##Preprocessing##
  Removing duplicates was essential for improving your validation set and consequently for finding  your class thresholds. To do so I compared image channels using the imagehash library proposed by  @tilii


##Base model##
  ResNet50 trained with float16 precision, RGB,  1024x1024, lots of data augmentation, 
  weight decay, sgd, gradient clipping, bce loss and multi-scale resolution images. All in PyTorch.
  Best single model [0.628 public, 0.553 private]

##Gamma correction##
 Due to the dark exposure of some images, half of my models were trained with an extra initial layer to correct the gamma. The goal was to find the best gamma per channel for all the images. I tried to make a layer that extracted a custom gamma per image taking into account the image as a context but without success. I tried things like  i) using statistical image values as features and ii) applying a small convolution to the image then fc; all of them very unstable. The approach that worked best was the simplest: learn independent gammas (y_1 , y_2 , y_3) per channel and only use one parameter initialized at 1 without an activation. This reduced my global val_loss and increased my internal F1_score of some classes.
  ![Gamma correction formula][1]
Amusingly all the learnt gammas were quite similar, in the range [0.6-0.65]. Meaning that the images were quite cleared. I guess dark images benefitted from this layers but others - specially noisy images - don't, that's why I trained also networks without it.



##What should I trust?##
Is my model overfitting? 
This was the big question of all the competition. It was hard answering it for the rare classes but for the other ones I found a pattern that I believe was quite useful. When comparing models, most of the times the models that were overfitting had the largest F1-Score in a zone where Precision was extremely dominating vs Recall.  Example in the following first figure. My hypothesis is that such models are learning things that are only present in the training set (similar cell/protein types, microscope features, who knows) and then become very confident in some samples increasing the Precision. When this happened I lowered thresholds or I discarded the model and choose another one (second figure).
![F1 curves][2]

##Model variety##
I trained 11 models, 2 DenseNets121 and 9 ResNets50. They were fast to train and provided the best results in my val_set.  Each model had different settings such as:
Excluding data leak, including it, change model seed, using yellow images , different class balance, full sized images, crops...

##Rare classes##
I did not know how to tackle them properly and I guess my fall has been mainly due to these classes (It's a pity that we cannot see F1-Score per class). I was about to try some Few Shot Learning approaches, but the image size was a limiting factor. Apart from this, I assume that the data leak eliminated most of the rare samples so detecting even just one of them was crucial (maybe too much). I hope I can learn from the top solutions about this.

##Ensembling##
I tried Logistic Regression, Averaging, Stacking and XGBoosting. In abundant classes it worked, but with others it overfitted and I thought it was not reliable. In the end I did the following:

*Expert models*
I decided which model were the best for specific classes, for instance: Yellow image model for classes 6, 7;   cropped image model for large classes such as 0, 25, full for 16; etc. These decisions were made taking into account hpa web info data, F1-curves, validation data, and public leaderboard.

*One vote ensembling with top predictions*
Apart from expert models I increased the recall of my predictions by adding the top predictions from other models. Most of my models scored similar F1-scores per class, so it made sense. I increased the thresholds of non-expert models using this heuristic formula T_opt = T_opt + (1 - T_opt) *0.3 and included them as in one vote is enough.


##Things that did not work##
- Training a Multi Head Attention Module (Transformer paper) from the predictions of crops of an image in order to focus the attention in the spatial domain. 
-  Training GapNet style/Feature Pyramid with ResNet18 as backbone did not provide any benefits in my setup (I guess I did not play enough with it because @Dieter was able to make it work).
- Training with 2048x2048 images, I guess batch size was too small and the extra resolution was not that important for the majority of the samples.

  [1]: https://i.postimg.cc/rmh5h58T/Captura-de-pantalla-2019-01-11-a-les-12-34-16.png
  [2]: https://i.postimg.cc/SR3cW84K/Captura-de-pantalla-2019-01-11-a-les-13-11-08.png
