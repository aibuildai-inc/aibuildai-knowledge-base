# 24th place solution - fast simple naive RNN

Competition: youtube8m-2019
Rank: #24
Source: https://www.kaggle.com/c/youtube8m-2019/discussion/112320

### 1. Train RNN on only validation set

I converted tfrecords into numpy array per https://www.kaggle.com/tbmoon/how-to-save-tfrecord-into-npy-too-slow to use pytorch.
Then I trained a 3-layers RNN only on validation set, there are 2 seperate GRUs for frame and audio data respectively, then concat them together and pass to a fully connected layer with softmax activation, this gave me 0.71 public lb. Also tried context gate, there is no much difference for this model.

### 2. Train RNN on train set and fine tune on validation set

For training set. I selected only samples with 1 label. To accelerate training speed, for each video randomly choose 15 continual frames for training. Train the RNN model on train set for about 4 hours, then fine tune on validation set.  This fine tune single 3 layers RNN model gave me lb 0.746. Trained another 4 layers RNN gave me lb0.735.  
The total training time for each model takes less than 6 hours on single P100 GPU.
My final submission is the average ensemble of the 2 models, which is 0.753 for public lb and 0.744 private lb.

### 3. Submission

I predicted every 5-seconds segment of test set, and use top 100000 scores of each classes for submission. The prediction takes 30 minutes on single GPU.

### Transformer / multi-task experiments
I replaced the RNN model with 3 layers or 6 layers Transformer, did not improve the score. Then I experimented mult-tasks on training set and validation set like this paper (https://arxiv.org/abs/1901.11504) , it was not hepful either.

I was also planning to experiment sequence to sequence learning with validation set, but was not able to complete the code before competition ends.
