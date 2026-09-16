# 45th Place | What I have learnt through TPS- Sep 2021

Competition: tabular-playground-series-sep-2021
Rank: #45
Source: https://www.kaggle.com/c/tabular-playground-series-sep-2021/discussion/275720

So, my first ever TPS competition has ended and I have finished at **45th** place at the final standing. Throughout this one month, I have learned a lot from the kaggle communities. I would like to appreciate all the kagglers who have shared their notebooks and discussed about different topics. These contributions helped me to improve my model gradually.
So, here I will summarize what I have learned through this competition.
- The main problem in this competition was **the dataset was huge** and there were **lots of missing values** in both the training and test sets. 
To tackle the huge dataset problem I sometimes used the "**DataTable**" library, but it didn't come out so efficient for me. Like , when you load your dataset with the "**DataTable**" library and again convert it to pandas with **to_pandas()** command, then the **binary values are automatically converted to True and False.** Like, 1's converted to True and 0's converted to False. That's why I had to convert this to again binary.
- **Decreasing the memory usage** came out very handy.
- Two times I ran out with GPU. I then tried to switch to Google Colab but there I faced problems working with LGBM. The error was,
**"LightGBMError: GPU Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_GPU=1"**
I used the following commands to overcome the problem.

> 
!rm -r /opt/conda/lib/python3.6/site-packages/lightgbm
!git clone --recursive https://github.com/Microsoft/LightGBM
!apt-get install -y -qq libboost-all-dev
%%bash
cd LightGBM
rm -r build
mkdir build
cd build
cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..
make -j$(nproc)
!cd LightGBM/python-package/;python3 setup.py install --precompile
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
!rm -r LightGBM


- As I had run out of GPU, then I had to write a code that can be run with CPU. But working with the CPU was taking a lot of time. That's why I first worked with **a less number of fold(K=2)**. If I saw the CV score is good enough, then I had chosen a large number of fold(**K=10 gave me the best result**) and clicked on the "save version" button, and ran the code. Then the code ran in the cloud and I didn't have to wait for the code to finish. 
- I worked with different types of folds (like, stratified k fold, kfold and repeated k fold). **Stratified k fold gave me the best results.**
- For imputing the missing value, I used simple imputer with the pipeline . And also scaled the values with **StandardScaler**. **Using "mean" gave good results than "median"**. I then save the processed result as a dataset. It saved a lot of time for me. As I didn't have to process the dataset all the time.
- Ensembling different models (like LGBM,XGBoost and CATBoost) gave good accuracy. **We can use the same algorithm for multiple types with different random state numbers.**
- **LGBM and XGBoost boosting algorithms can handle missing values** whereas CATBoost algorithm gives errors.
- **Tracking the hyperparameters and their respective outputs is necessary**. It helps you to understand what type of change you need to make to get a good accuracy. For this, [neptune.ai](https://neptune.ai/) can be used. It will help you in **organizing your work, tracking, and visualizing** your experiments.
If you want to use this service in your code, you can check out this[ link](https://towardsdatascience.com/track-and-organize-your-ml-projects-e44e6c7c3f9d).
- In the end, ensembling the outputs with **weighted average and powered submission** can give you an increase in the accuracy.
