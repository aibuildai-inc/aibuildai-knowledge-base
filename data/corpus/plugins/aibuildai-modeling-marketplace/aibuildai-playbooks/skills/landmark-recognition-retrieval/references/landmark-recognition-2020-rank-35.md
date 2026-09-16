# (Public>0.54) Provisional 36th solution and kernel

Competition: landmark-recognition-2020
Rank: #35
Source: https://www.kaggle.com/c/landmark-recognition-2020/discussion/187714

Thank Kaggle for hosting this awesome competition! And congrats the top teams!
This link is our kernel (version 10) 
[https://www.kaggle.com/tangshuyun/lb-0-54-effnetb6b7-global-feature](url)
Our approach is very straightforward and simple:
Thanks to Ragnar's awesome kernel of training Effnet, we find some practical ways to train our own models. After reading Keetar's fantastic writeup of his GLD retrieval, we trained our Effnet B6 and B7 first with 384 sized images. Then we use the increasing 512 sized images to further tuned our B6 and B7. The training environment is Colab Pro. Then we simply put the two model ensembling predictions to the global feature extraction.

For more details and adjustments, please refer to the kernel above. 

Feel free to ask if you have any questions
