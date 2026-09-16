# 22nd Place Solution: Just Encoder Modules

Competition: riiid-test-answer-prediction
Rank: #22
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209711

Thanks for all the discussion and support from fellow Kagglers. I've learned a lot in the competition and tried a lot of things. Here a simple list of the stuff that I believe worked well. I'll try to cover the whole model but may make updates to clarify and add explanations later. 

**[Link to notebook ](https://www.kaggle.com/abdurrafae/22nd-solution-just-encoder-blocks)**

**Architecture**
My architecture was just 3 Encoder modules stacked together using 512 as d_model and 4 Encoder layers in each block.

My Questions, Interactions and Response sequences were all padded on the left by a unique vector made using historical stats from the user. This ensured that all 3 sequences were aligned on the sequence number


1. Questions Only Block (Q-Block):
This one was just self attention over the questions.

2. Interactions Only Block (I-Block):
This one was just self attention over the interactions.

3. Questions, Responses, Interaction Block (QRI Block):
I used output from Q-Block as query, I-Block as keys and Responses/(QRI Block) as values. The residual connection after attention module was made using value vector (instead of the default query vector). Self Attention Mask was used in first 3 layers and for the last layer I used a custom mask that only attended to prior interactions and didn't have the residual connection.

Concatenated the output of Q-Block and QRI-Block, feed it into 3 Linear layers.

**Data Usage**
I split the longer sequences into windows of 256 length and 128 overlap. (0-256, 128-384, 256-512).
I saved them in tf.records and during inference I took a uniform random 128 window from each 256 length sequences. This ensures any event would be placed at 0-127 location in my training sequence with equal probability (excluding the last 128 events in the last window of a user). For users having smaller sequence length I padded with 0 to make the length equal 128 and used those 128 each time.

**Compute Resources**
I initially only used Kaggle GPUs to train the model. Started setting up a TPU pipeline in the last 3 weeks of the competition and in the end was able to use the TPUs as well. I think I've exhausted my complete TPU quota for last 3 weeks.

I'd like to thank @yihdarshieh (**TPU Guru**) for this [TPU notebook](https://www.kaggle.com/yihdarshieh/tpu-track-knowledge-states-of-1m-students). It helped a lot in setting up the TPU pipeline.

**Sequential Encodings**
I used 2 encodings that captured the sequence of events. I subtracted the first value in each sequence from the rest to ensure that all encodings started with a 0.

**Temporal Encoding**
For this I converted the timestamp into minutes and used the same scheme as that of positional encoding with power (60 x 24 x 365 = 1 year). I believe this enabled the model to know how far apart in time were 2 questions/interactions. I believe this to be a better implementation of the lag time variable used in SAINT+ as it captures the difference in time between all of the events simultaneously rather than just between 2 adjacent events.

**Positional Encoding**
For this I used the task_container_id as the position and a power of 10,000.


**Proxy for Knowledge**
Instead of just using 0/1 from responses I added an new feature which is a heuristic for knowledge in case the response was incorrect. If a user selects option 2 when 1 is the correct one. I would calculate (total number of times 2 was selected for that questions)/(total number of times the question has been answered incorrectly). This ratio was maintained and updated during inference as well. I checked the correlation of mean of proxy_knowledge for incorrect answers and overall user accuracy, it was around 0.4.


**Question difficulty**
Simple feature that is calculated as  (total number of times the question has been answered correctly)/(total number of times the question has been answered). This ratio was maintained and updated during inference as well.


**Custom Masks**
I used self attention mask that didn't attend on events of the same bundle.
For the last layer of QRI block I removed the mask entries along the diagonal to ensure it only attends to prior values.

**Starting Vector**
For the starting vector I used counts of the time each tag was seen/answered correctly in questions and lectures. Just used a dense layer to encode this into the first vector of the sequence. 

**Other Details**
- Used Noam LR provided on the Transformers page on TF documentation. 
- Batch size 1024 (Trained on TPUs - 1 Epoch took 4-6 mins)
- For Validation I just separated around 3.4% of users initially and used their sequences.
- Model converged around 20-30 Epochs. (4 Hours training time at max)
- Sequence length of 128 was used.

**Question Embeddings**
I added up embeddings for content id, part id, (a weighted average of) tags ids and type_of (from lectures), then concatenated it with both Sequential embeddings and then into a dense layer with d_model dimensions.


**Response Embeddings**
I concatenated answered correctly, proxy knowledge, question difficulty, time elapsed and question had explanation and feed into a dense layer with d_model dimensions.

**Interaction Embeddings**
I just took the first d_model//2 units from both Questions/Response embeddings and concatenated them for this.
