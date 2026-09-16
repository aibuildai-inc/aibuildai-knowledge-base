# [12th solution] Full, reproducible solution with code

Competition: asl-fingerspelling
Rank: #12
Source: https://www.kaggle.com/c/asl-fingerspelling/discussion/436457

I already wrote the gist of my solution [here](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434363). Here, I will give a more comprehensive summary that follows official writeup guidelines. I also finished cleaning and arranging my work [in my github](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution). I made efforts to make my solution as easy to follow and reproduce as possible. I saved everything as Colab or Kaggle notebooks that you should be able to run as is. I kept all the data in public Kaggle datasets. Please let me know if I missed anything; I will do my best to fix it. If anything needs to be clarified, ask me. 

### Context section
This is a summary of my 12th place solution to [Google - American Sign Language Fingerspelling Recognition competition](https://www.kaggle.com/competitions/asl-fingerspelling/overview).
The data page is [here](https://wwww.kaggle.com/competitions/asl-fingerspelling/data).

### Overview of the Approach
My model is a CTC encoder with transformers and convolution layers, the same as the one used in the previous competition [1st solution](https://www.kaggle.com/competitions/asl-signs/discussion/406978) with the CTC modification that was introduced [by Rohith](https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place). My only modification to the model (besides changing the size) is adding positional encoding right before the first transformer layer.
For the features, I used a similar approach as the solution of the previous competition and took the same lips, nose, and hands landmarks. I did not use the eyes and used slightly different pose landmarks (added some). I used X and Y (without Z) and also used difference features with a skip of 1 and 2, also the same as the previous competition solution.
For augmentation, I used the same augmentations as the previous competition solution, with a modification to all augmentations based on modifying/deleting data in a window. The original solution used a random window, but a random window give a lower probability of modifications in the edges. I changed it to a random circular/rolling window, i.e., instead of ignoring the window section that goes out of the frames length, I rolled it to the beginning of the sequencs and modified it too.
For filtering the bad data, I used two-step filtering. In the first step, I filtered the data according to (number of frames with non-nan hands landmarks) > 2*(length of phrase) and trained a basic model. In the second step, I used the basic model to calculate the normalized Levenshtein distance scores of all the samples and filtered them according to the score with a threshold of >0.2. This filtering allowed me to add to my training a lot of samples that I filtered out in the first step.
A 9,497,577 parameters model with the previous competition solution's augmentations trained on ~300 epochs got a public LB of 0.779. making it larger (15,471,112 parameters) and training for 500 epochs got a public LB score of 7.9. Adding the modifications to augmentations and second-step filtering and training for 1500 epochs got public LB  0.794 and private LB 0.779 (the final, 12th-place solution).
For validation, I used the first 3,000 samples. Toward the end of the competition, I feared that I might be overfitting to existing signers, so I did one experiment with 1000 epochs and separated the validation fold by signers ID (I used five unique signers for validation with training on all the rest). Even after 1000 epochs, the scores on the validation fold did not suffer due to overfitting. Thus, I continued with the original validation since I preferred to train on samples from all the signers.
I did most of my training on Colab TPU. They are cheap and easy to use with kaggle datasets (no storing or egress costs).
### Details of the submission
Besides what I wrote in the previous section, I had to change the maximum frame number after finishing the training to 320 since my original 340-frame model could not complete the inference in time. Of course, I did previous experiments and expected the 340 max frames number model to complete the inference in time. I suspect that with more epochs, the complexity of the parameters is higher, leading to more complex quantization and, hence, more inference time. Of course, I had to quantize to 16-bit for my model to fit the 40MB limit. Also, I could not train it in 16-bit since I used TPU with their bfloats. However, the quantization did not have a significant effect on my scores. If it had any impact, it was less than 0.001.
What did not work: mix-up augmentation and AWP, but with more time, I probably could have made them work, too. I just made some blunders at the beginning, and when I understood how to do things right, there was already not enough time to train new models.

### Sources
As I wrote above, a lot of my work was based on:
1. [previous competition 1st solution](https://www.kaggle.com/competitions/asl-signs/discussion/406978)
2. [Rohith's notebook](https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place) (in particular, the CTC modification and TFlite submission code)  

My cleaned and reproducible solution can be found [on my GitHub](https://github.com/shlomoron/Google---American-Sign-Language-Fingerspelling-Recognition-12th-place-solution).

It was a great competition, and I enjoyed it a lot. Thank you, Google, and everyone who published code or participated in forum and code section discussions. You made this competition as approachable and enjoyable as it was.
