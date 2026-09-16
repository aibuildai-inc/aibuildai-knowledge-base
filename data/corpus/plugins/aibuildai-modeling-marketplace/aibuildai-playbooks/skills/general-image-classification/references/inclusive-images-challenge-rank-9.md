# 9th Place - Solution

Competition: inclusive-images-challenge
Rank: #9
Source: https://www.kaggle.com/c/inclusive-images-challenge/discussion/71525

My solution is simple. 
I downloaded only train_00.zip and train_08.zip due internet speed restrictions.

Merging datasets brings more than 343,042 images and 1,983,191 labels associated. Then I dropped labels with frequency less than 50 decreasing the number of labels to around ~ 1.6M.  This way the unique number of labels in my dataset decreased to 3862.

I used Keras NASNetLarge to train the model from scratch using 224x224x3 images. During training I monitored a custom f@2 metric after each epoch.

Augmentation was made using albumentations lib:

    def augment_flips_color(p=.5):
      return Compose([
          CLAHE(),
          HorizontalFlip(.5),
          ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=10, p=.75 ),
          Blur(blur_limit=3, p=.33),
          OpticalDistortion(p=.33),
          GridDistortion(p=.33),
          HueSaturationValue(p=.33)
      ], p=p)

Model trained for several epochs until I got a f@2 of around 0.63 in one small validation set.
Using 4xTTA my validation improved to around 0.64 only.

Searching for the best threshold to cut the predicted labels I found the value 0.049 that brings f@2 to 0.68 in the validation set. It means my final solution predicts all categories with prob &gt;= 0.049 for each image.

I tried the same approach using a Xception and a VGG-16 architecture. Both scored less than the NASNet, but when I blended the three models in stage 1, it improved the LB score around +0.01. For stage 2 I used only the NASNet in the solution.

Giba
