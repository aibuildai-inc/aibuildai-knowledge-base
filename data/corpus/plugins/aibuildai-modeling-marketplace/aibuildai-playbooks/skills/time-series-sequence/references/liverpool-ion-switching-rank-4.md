# 4th place solution

Competition: liverpool-ion-switching
Rank: #4
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153911

First, I would like to thank Kaggle and the host @richardbj for organizing this competition. I find the competitions with scientific and mathematical background the most interesting. Of course a leak is always bitter.
The solution I created is the average of 4 very similar Keras models. They are based on encoding the signals based on the results of Gaussian Mixture Models, initial processing (head) using Dense and BiGRU layers for feature creation, a Wavenet Model for main processing and final steps creating the classification using Dense layers. Additionally two tree models based on Random Forest Classifier (RFC) and Histogram-based Gradient Boosting Classification (HGB) are used as stacked input. 
Following is a description of the individual parts

## Data Preparation and Preprocessing
The data is separated into batches of 500000 records each. This is done with the dataset df (for train and test) through df.batch = df.index // 500000. In the same manner a split into  segments of 100000 records and windows of 10000 records are created. 

Initially the segments can be classified into segment types 1,...5 based on the values of mean standard deviation over the segments. 

As identifed by @cdeotte the signals contain a sinusoidal drift. This drift is removed automatically for train and test data using a python script. This is done by fitting and subtracting a sinusoidal curve from the signal over a complete batch and then a segment if not possible for the batch. 

The important part here as well as for the definition of segment types is to do this automatically without any visual inspection and being able to execute the scripts on new test data. Otherwise the solution would be invalid.

The resulting signals are visualized below as discussed in the forum already. For the train data with batch number

and the test data with segment id.


The training data contains outliers in batch 8. I did not see any way of correcting this and therefore cut this out of the training data.
When taking the signal values for each open channel and building the mean for each window one gets an interesting plot

This shows that the signal strengths for an open channels value are approximately equal for segment types 1, 2, 3 and 4, but shifted for segment type 5. 
I evaluated the possibility to use the segment type for models and even split models. I came to the conclusion that this would be an error and could overfit badly. Instead I grouped segment type 1, 2, 3 and 4 into a segment group 0 and segment type 5 into segment group 1 and used the segment group as a feature in modeling.

## Gaussian Mixture Models
The target variable Y = number of open channels is based on a HMM. The signal X is a function of Y, but including a Gaussian Noise. Evaluating the conditional distribution P(X|Y=k, Segment) gives Normal distributions with mean mu\_k = mu\_0 + k*delta, but with standard deviation that differ slightly between segment types and are larger for segment group 1. The image below show an approximation of the signal through GMM for one sample segment

The red curve shows the weighted sum of normal distributions with center corresponding to the discrete mean values (vertical lines) associated to the open channels value and the weight corresponding to the fraction of target values equal to that open channels value.
## Signal Encoding 
The modeling of the signals as GMM enable encoding the signal X as a vector (f\_k),  k =0,…,10 using the normal distribution density function N(mu, sigma)(X), with mean mu_k and sigma depending on the segment type. 
This encoding of the signals is one key input into the models.
## Models
The main models are based on Keras Wavenet models with specific head and tail processing. 
Initially I focused on sequence processing using BiGRU, but they are much slower and therefore did not allow as much experimentation. I did not manage to get a better score that 0.942 on the public LB. 
When changing to Wavenet I initially used RFC and HGB models as additional input, but also the already created Keras models with BiGRU as stack input. This reached 0.946 on the public LB, but I had the impression it would overfit. I therefore dropped the BiGRU models as stack input and included BiGRU as a processing step together with the Wavenet model. 
The local CV results actually dropped, but the LB results increased and inspecting the predictions gave me seemingly better results. This gave me the final models.
For the Keras models I did not use early stopping. I saved 3 models: With best f1 score, with lowest categorical cross entropy and latest (for stacking purposes). When initially testing on the LB the one  stored based on “best f1 score” always had better LB score than the “lowest loss”. I therefore sticked to that. 
The Keras models were trained using Adam optimizer and SWA as well as a One Cycle Scheduler with max learning rate of 1.0e-3 and 300 – 350 epochs. 
## Cross Validation
For validation I used 5 fold CV. Prior to splitting the train data into train and validation data I split it into sequences of 4000 records and used StratifiedKFold on the batch as group. This makes certain that the batches are distributed equally among the folds. But what about the target variable? For this I looked at the distribution of the target variable for different random state values. The distribution depends strongly on that value and I looked for a value giving similar distribution of each class for both train and validation folds. So which one to choose? The answer is not 42, but one good answer was 43 (also 53 or 132). 
For the tree models for stacked input it is important not to use the validation folds for early stopping, but only for storing the out of fold predictions. Both RFC and HGB support this directly. I also tried Catboost with cross validation for determining the number of iteration, but did not use this  in the final models. I was not able to get anything sensible out of LightGBM and (my version of) xgboost hat errors that did not allow me to use it.

## Analysing Predictions on Test Data
The CV results turned out to correlate very well with the LB score, but there were potential issues in particular with the different test data for segment type 1. For train data these only had open channels values 0 or 1, but for test data it turned out to have higher values 2, 3, or even 4. 

For validating this I created scatter plots of the signals with predicted open channels values for each window (every 10000 records) showing horizontal lines for the relevant mean values from the GMM models. The plot below shows two windows 


Looking at these different windows one can validate if there are “obvious” errors and try to enhance the models. When using Wavenet model alone I saw such errors that were then corrected when adding a BiGRU layer.

## Some Final Words
I see the models as a good basis for further work. I had limited hardware resources (single GTX 1080Ti)  and could not analyze variants as wished. It can definitely be improved.
Finally: It is unfortunate that a leak occurred and partly spoiled the results.
