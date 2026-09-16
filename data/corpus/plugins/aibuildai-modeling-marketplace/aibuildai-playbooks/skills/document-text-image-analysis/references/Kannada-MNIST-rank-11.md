# 11th place solution

Competition: Kannada-MNIST
Rank: #11
Source: https://www.kaggle.com/c/Kannada-MNIST/discussion/122158

*Checkout my kernel at :-*
[[KMNIST__top__1%]](https://www.kaggle.com/namanj27/how-i-landed-up-in-top-1)

**Basic Intro**
This was my first ever general competition. I was already in top 1% in my InClass competition ( MNIST ) so thought of giving KMNIST also a try.

***Small Things I did that *helped* me bit by bit***:-

1.**Mean Normalization and feature scaling**

2.**.astype('float32')**
Converted xtrain, ytrain, xval, yval to float type

3.**LeakyReLU**
Used LeakyReLU insted of our good ol' ReLU. Saw drastic difference in accuracy.


***My OBSERVATIONS***
1.**Ensembling** { DID NOT HELP }
I tried two approaches here. *First approach* I ensembled various different models and took their average output followed my argmax. *Second Approach* I ensembled same model multiple times and took their average output followed my argmax. Neither of them helped me.

2.**Training on Dig_MNIST** { DID NOT HELP}
I combined both *train* and *Dig-MNIST* data and tried to train my model on the combined data. But it was of no use 👀 

3.**Optimizer** { HELPED }
After trying out multiple optimizers I finally chose RMSprop as the better optimizer for this competition.

4.**Image_Data_Generator** { HELPED }
Got a big jump from 0.96 to 0.98 public score using tweeked hyperparameters of Image_Data_Generator class along with *learning_rate_reduction* and *earlystopping*.

5.**MOST IMPORTANT LEARNING** { HELPED }
*Hyperparameter tuning*💯 
Once I got around 0.98 public score after applying all those awesome techniques ( stated above ) I used my effort to tweek all those hyperparameters ranging from *epochs*, *batch_size* ... to *rotation_range*.
