# 8th Place Solution /w Code (minimal ver.)

Competition: bengaliai-cv19
Rank: #8
Source: https://www.kaggle.com/c/bengaliai-cv19/discussion/135990

Hi, everyone

First of all, I want to thank Kaggle and the hosts for hosting the competition.

Congratulation to all winners!




An illustration of my pipeline. It contains 2 models, Seen Model and Unseen Model.

# Summary

* NO EXTERNAL DATA

* I used 2-stage prediction in my pipeline, it contains 2 models as one set (Seen Model and Unseen Model) as shown in the figure above.
* I used Arcface to distinguish unseen graphemes. Just like to distinguish unseen faces ;)
* If an image is detected to be seen, I use the output of Seen Model as prediction directly. Else if the image is detected to be unseen, I’ll pass it to Unseen Model to get unseen prediction.
* The different points between Seen Model and Unseen Model are as follow:
  * Seen Model only uses very few augmentation to make sure it can ‘overfit’ to seen graphemes, while Unseen Model uses heavy augmentations to make it generalize to unseen ones.
  * Seen Model has 5-head outputs, including Arcface output. While Unseen Model has normal 4-head outputs.
  * Beside the number of output heads, the architecture of the top is a little bit different as well.
  * I decompose the predicted grapheme to 3 components when the prediction is made by Seen Model, while I use predicted 3 components directly when the prediction is made by Unseen Model.


# Code

I’ve published a minimal version of my training &amp; testing pipeline on Kaggle kernel as follow:

* Step 1. [Bengali Train Seen Model](https://www.kaggle.com/haqishen/bengali-train-seen-model) (trained ~5h on kernel)
* Step 2. [Bengali Train Unseen Model](https://www.kaggle.com/haqishen/bengali-train-unseen-model) (trained ~8.5h on kernel)
* Step 3. [Bengali Predict with Seen &amp; Unseen Models](https://www.kaggle.com/haqishen/bengali-predict-with-seen-unseen-models) (submission ~20min)

The minimal version with two efficientnet-b1 which are training on 128x128 for 30 epochs can give you public LB 0.985+ or private LB 0.935+ 
If you find my notebooks helpful, please upvote them! 

---

To get over 0.993+ or 0.950+ on public or private  LB, what you need to do is just simply:
* Change the backbone to efficientnet-b5 / b6 / b7 / b8
* Train on 224x224
* Train for 60~90 epochs
* Emsenble more Seen &amp; Unseen Model

If you have any questions please feel free to leave a message to me.
Thanks!
