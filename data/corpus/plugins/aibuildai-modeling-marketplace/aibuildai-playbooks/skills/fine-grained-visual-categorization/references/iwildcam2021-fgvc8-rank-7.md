# 7th Place Solution

Competition: iwildcam2021-fgvc8
Rank: #7
Source: https://www.kaggle.com/c/iwildcam2021-fgvc8/discussion/242978

I created a mini dataset (10% of the original dataset) with the same class proportion as in the original. And carried out many experiments to move quickly through the competition.

1) Preprocessing -
In my initial experiments, I found that instead of directly cropping and resizing the detections, cropping and padding the detections using reflection padding improved the score. Hence my final solution used reflection padding while training and testing. My preprocessing pipeline was as follows

crop detection -> reflection padding -> augmentations -> standardize

2) Augmentations -
In my experiments adding any type of augmentation reduced my validation score. Hence I kept simple augmentations with very little intensity. The augmentations that I used were as follows 

Random Horizontal Flip (p=0.5)
Color Jitter : Brightness and Contrast (0.9 to 1.2)
Grayscale (p=0.5)
Gaussian Blur (sigma = 0.0 to 0.8)
 
3) Handle Imbalance - 
Here is how I have handled imbalance: https://www.kaggle.com/c/iwildcam2021-fgvc8/discussion/242455

4) Model Architecture -
For my mini dataset, I experimented with efficient b2 noisy student. For the final submission, I used the efficient b5 noisy student. I did not use pre-trained image net weights.

5) Submission pipeline -
In a public kernel, I came across the max count logic. In this, the maximum number of detections from an image in a sequence (max count) was considered as the prediction value for that sequence. I modified this logic and also considered maximum frequency. For eg - in a sequence of 9 images having only one specie of animal, the number of detections for each image were - [1,1,1,1,2,2,2,4,2]. So the max count logic will predict 4 while max count + max frequency logic will predict 2. There might be a strong possibility of false detections, and thus image number 8 having 4 detections (out of which 2 could be false detections) would be eliminated. The frequency of both 1 and 2 is the same, so the max count will be considered, that is 2. I saw a validation accuracy boost and public score boost using max count + max frequency logic, and hence used in my final submission.

Also, a higher megadetector detection confidence threshold worked better, so I used 0.7 as detection confidence for the final submission pipeline.

6) Training -
I experimented with mixed-precision training, but training without mixed-precision yielded better results for me. I did not use any LR scheduler for mini dataset experiments. But for the final training of efficient b5 noisy student, I used reduce on plateau. I underestimated the training time required for the efficient b5 noisy student. I spent all remaining 26 hrs of GPU in the last week in efficient b5 noisy student's training and still, it was a little underfit with ~ 78% val accuracy and 82% train accuracy. After exhausting my Kaggle GPU quota I switched to colab and resumed training. This time I changed some hyperparameters like batch-size and LR, but I could only reach 82% val accuracy and 85% train accuracy due to lack of time (deadline was near). But I couldn't make a submission using colab because I later realized that the test set was 30GB while on colab I had 28GB disk space. I couldn't even download the zip there. So my final model is still underfit (i.e. it needs more training).  

**Here is my final submission pipeline [notebook](https://www.kaggle.com/devashishprasad/iwildcam2021-submission)**
