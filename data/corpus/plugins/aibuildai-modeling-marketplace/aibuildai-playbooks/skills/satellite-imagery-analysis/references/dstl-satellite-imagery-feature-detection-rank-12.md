# 0.53 Public LB Solution

Competition: dstl-satellite-imagery-feature-detection
Rank: #12
Source: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/discussion/29790

# Summary of the DSTL Challenge
In this document I will review all my work in this challenge

## 1. First steps
### Preprocessing the images
When I looked at the rgb images I noticed that the white balance was no good. So I applied white balance like in the link below.  
https://docs.gimp.org/en/gimp-layer-white-balance.html  
Instead of applying white balance to single images I made collages(for example all the images that start by 6010) and I applied white balance to the whole collage. I made this for all the satellite channels.
I also corrected the missalignement between the different channels using simple traslations.
### First trainings
At the beginning I tried to train a single model to predict all the classes. I found it very difficult and I think it was because the great imbalance between the classes.  
![enter image description here][1]   
I decided that I was going to train a different model for each class, I tried that and it worked better so I continue with that aproach until the end.
### Sampling 
The low frequency of the classes make hard to train them. So I decided that instead of training each epoch with all the samples I was going to take some samples with positive instances and some samples without. That way I could change the frequency of the classes and help the model to learn.
### Creating the submission
This was probably the most painfull part of the challenge. The slow evaluation method of kaggle forced me to simplify the predictions. I could not make a succesfull submission for trees without simplifying.   
I used shapely for making the simplifications, and at the start I was simplifying without preserving the topology so the scores were badly hurt. I discovered this when I made a tool for visualizing submissions, I was just visualizing predictions so this shows how important is to visualize each step of the pipeline.  
My last submission was made with tolerance 8e-6 ~ 4 pixels.

## 2. Tools
I have used python and the main libraries used are:
* Keras
* Opencv
* Shapely  

I started the challenge using a laptop with a 980m gpu and 8GB of RAM. At the half of the challenge I bought a pc with a gpu 1080 and 32GB of RAM, and in the last week of the challenge I bought another gpu 1080 and another 32GB of RAM :)

## 3. Models
I think this competition was pleasing because it was really nice to watch how the model was learning to paint a map just by using the images. Below you can find some visualizations of images from the test set.   
 
![enter image description here][2]
![enter image description here][3]
![enter image description here][4]

Main characteristics of the models   

* I used unet architecture on all my models.   
* I trained using cross-validation with 3 and 5 folds.  
* For each class I tuned the depth of the model, the window size and the resize of the images.  
* I used as input all satellite bands except for the models with finer detail. In the case of structures and vehicles I only used rgb images and p band.  

![enter image description here][5]

### Water
This is my most worked model. At the beginning I tried to train a model for rivers and another for lakes. But then I realized that it was really hard for a model to segment the image pixel by pixel and also make an abstraction and decide if something is a lake or a river.  

So I decided to train a model just for predicting water, and after that add a high level classifier to make the division between lakes and rivers.

Another idea was to use images with different scale for training the model. I realized that unlike the other classes a lake or a river can have all sort of shapes and sizes (all cars will have a similar size). So I trained the model with images with scale 1:1, 1/2, 1/4 and 1/8. That way my training set was much bigger.

Before feeding the images to the high level classifier I made collages with them because sometimes in a single image we don't have enought information to decide to which category they belong.   

![enter image description here][6]
The high level classifier had only two steps: in the first I use a threshold on the contour size to search for big rivers. In the second I use the distance to the big rivers to decide if a contour was a lake or a river. 

Having a good water model was important, because later I used this model to clean the output of some of the others.



### Building, road, tree, crops, track
I don't think this models have something special. I just tried with different parameters such as: resize, window size, depth, sampling until I find a good configuration.

### Structure
This model was hard to train. I had to use extracost of 2 or 4 on white pixels to force the model to learn. Otherwise it painted all black.

### Small and big vehicles
I spend the last days of the contest trying to improve this two categories.  
My final submission consisted on averaging the prediction of a lot of models (up to 15 different models). For trainign different models I used cross-validation with different random seeds.  
I cleaned the prediction using the prediction of the other models. I used the water, building and tree model for cleaning.


## 4. Scores
![enter image description here][7]


Please let me now if something is not clear or if I have to elaborate more  on some topic.  
I will be adding more info if necessary to the first post.   

Thanks to the organizers and the participants.  
ironbar


  [1]: http://imgur.com/exlz22p.png
  [2]: http://imgur.com/1vdZ5em.png
  [3]: http://imgur.com/8jPC2UM.png
  [4]: http://imgur.com/R5QQiJG.png
  [5]: http://imgur.com/7LHvaOr.png
  [6]: http://imgur.com/9K6XUdg.png
  [7]: http://imgur.com/9EJggch.png
