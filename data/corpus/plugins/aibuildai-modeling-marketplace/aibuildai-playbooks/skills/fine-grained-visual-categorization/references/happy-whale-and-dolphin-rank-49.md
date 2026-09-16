# 49 place  - Silver Solution

Competition: happy-whale-and-dolphin
Rank: #49
Source: https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/319911

First of all, I want to thank Kaggle for the super platform and super community. Everything I learned about DS I learned on Kaggle. Secondly, thanks to the organizers of the contest for an interesting task, it helped me get distracted from sad thoughts.

**Dataset**
The key point in the competition was the dataset. I used **Baskfins** and **Fullbody** datasets by JAN BRE
There was an idea to use an intermediate dataset, but I didn’t have time.
I also tried a dataset with water removed, but the result was much worse. I added it in ensemble.

**Model**
I tried Effnet models and Effnet 7 work better.
ArcFace loss.
Image resolution is 768. I tried 1024, but model does not converge(
I also tried ConvNext result was worse, but it added in ensemble.
30-50 epochs with 5 epochs warm-up and then cosine decrease.
There was no overfit at all. More epochs - better results )

**CV and Ensemble**
I connected embeddings from 9 models and found the cosine distance between train and test images.
Since there was a lot of data and very little time at the TPU, I did cross-validation on only 1 fold. 
And get 0.72 CV without **new_individual** and CV 0.86 with threshold 0.6
And then trained all models on 100% train data, and use threshold 0.65 since in test dataset less **new_individual**

*Thanks for reading!*
