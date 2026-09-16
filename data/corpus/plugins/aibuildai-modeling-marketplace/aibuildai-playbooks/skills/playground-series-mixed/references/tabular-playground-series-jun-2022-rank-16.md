# #16 Deep Magic

Competition: tabular-playground-series-jun-2022
Rank: #16
Source: https://www.kaggle.com/c/tabular-playground-series-jun-2022/discussion/334358

It was a good morning June 18, I had passed all my exams and thought I would take some time off from the endless events going on in the world. I noticed another Tabular Playground competition and thought, "Why not? I'll give it a try."
So, after thinking about the task for a while, I decided to just cycle through all of NaN using Catboost. I got 0.89131 Public score (and 0.88957 Private). If anyone is interested in the Catboost solution, I used parameters **lr = 0.6222, depth = 4, n_estimators = 12000, leaf_estimation_method = "Newton", leaf_estimation_iterations = 10, posterior_sampling = True, l2_leaf_reg=7, random_strength = 3 with RMSE metric**. Not bad, fast and pretty accurate. But I was still unhappy with my result, so I decided to try neural networks. 
At first I used PyTorch with heavy architecture (256-128-64... etc) and Batch Normalization on layers. I got a score of 0.83784 on Public (0.83593 on Private). I was happy, but this solution didn't run on Kaggle because of execution time limits, I had to run one notebook with nan count = 1, then another notebook with nan count = 2 and so on, because the execution time was just huge - I killed two weeks for that. So I decided to implement a deep neural network, but with fewer weights, this time on Keras. 

On all layers, except the output Dense, I used the **swish activation** (mish, by the way, in this task is much worse). As optimizer I used AdamW with the following parameters: **lr = 0.012, amsgrad=True, weight_decay=4e-7, beta_1=0.95**. I didn't use any callbacks because it reduced performance and time was important in this task. The number of epochs was 70, batch size = 16384. 
RAdam actually showed better accuracy, but I simply could not use it as it was much slower and would not allow the solution to fit in 36000 seconds. Using 300 epochs and RAdam would have given amazing results, but my personal AMD GPU told me "No! 😐 Go buy NVIDIA with CUDA".
By the way, I didn't even have time to upload my last solution using **lecun_uniform as a kernel_initializer**. You can try it, it also improves accuracy. 90% of all my time was spent training models and optimizing them within runtime/precision. Of course, I am by no means blaming Kaggle for anything, but, I think, many participants had a hard time with this task due to notebook time limits.
**And... Of course, congratulations to all the winners and thanks to Kaggle for new and interesting challenges that allow me to get better.**
Good luck! 😊
