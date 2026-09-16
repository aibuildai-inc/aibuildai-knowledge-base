# 5th Place Solution Sharing: A Learnable Pooling Approach

Competition: landmark-recognition-2021
Rank: #5
Source: https://www.kaggle.com/c/landmark-recognition-2021/discussion/275876

Congratulations to the winners and thanks google and kaggle for hosting such an interesting competition! 

This is my first time to participate in the landmark recognition/retrieval competition. After reading the previous winning solutions, I also felt the anxiety with the limited computation resource I have: 1x3090 at the very beginning. So I decided to find a better(affordable) model architecture for the task, instead of following the common network+gem+arc_face approach.

Based on the observation that larger input images would lead to better accuracy, I tried to use the intermediate feature maps from the backbones. With my experience in the Yourtube8M video classification, I then find that NeXtVLAD(https://arxiv.org/pdf/1811.05014.pdf) perform pretty good in aggregating and merging features from these different feature maps. 


### Model Architecture 



As you can see, each feature map is aggregated to one high-dimensional 1D feature with NeXtVLAD, then those features are concatenated and fed into a simple linear projection layer with output_dim=512. In my implementation, I also add more non-linearity with another SE-gating layer(which was actually found to be not really helpful). At last, we use the arc face product and dynamic margin for the loss.

Training setup:
V2X: 3.2M images with 81313 landmark
V2C: 1.6M images with 81313 landmark
V2Full: 4.1M images with 203094 landmarks+ nonlandmark
 
1) fix the weight in backbone and only train the learnable pooling layer with V2X for 10 epochs, which surprisingly have very high local validation(around 90%). [AdamW, 0.001 initial, 0.8 decay)

2) finetune the whole model with V2C for 10 epochs(around 94% local validation accuracy)[AdamW, 0.0001 initial, 0.9 decay)

3) change only the arc_face_product layer with 203095 classes(use the nonlandmark from 2019 testset is really helpful). Again, fix the weight of backbone and only tune the learnable pooling layers with V2Full. (around 95-96% accuracy after 10 epochs in local validation)[AdamW, 0.001 initial, 0.8 decay]

The advantage of the learnable pooling layer is:  

1. The training is super fast as you only need to train the learnable pooling layers at step 1 and step 3.
2. I use the image input size of 256x256 at step 1 and step 2. Then increase the image size only at step 3, when the backbone is fixed(so I never need to train an efficientnet with large input images).

For instance, training a swin-base-224 only takes around 3 days using single 3090 GPU with the setup. 

I achieved top10 around early Sep with just one 3090. And then my teammate @marbury provided me with another 2xV100 so that I can just scale to different backbones. My final solution is an ensemble of
- swin-base-224
- swin-base-384
- swin-large-224
- resnest101 (512 input at the step 3)
- efficientv2-m (512 input at step 3)

The final LB score is highly correlated with the accuracy of the backbone models in imagenet, which I suppose show me that this learnable pooling approach have good ability to do the transfer learning. 

It would be interesting to see what the model arch can achieve with more complex backbones, including efficientb5-b7, efficientv2-large and swin-large-384.  There is still plenty of room to improve and hopefully to inspire more ideas & works using the learnable pooling module for the competition in next year. Will share my code once it is cleaned up.

### Other Insights
- using the whole v2full as the index set during the inference is important to stay at top 10.
