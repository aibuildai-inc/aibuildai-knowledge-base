# 2nd Place Solution

Competition: playground-series-s3e5
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/388011

Hi Everyone!  😎

First, I appriciate the Kaggle team for the competition.  
I was suprized that I got 2nd plaice in private score board only a month and a half after I started participating in the competition. 
With work and raising a family, I don't have much time.The two weeks available for this competition allowed me to think carefully and create a model.  
I learned a lot through this competition.　Arigatogozaimasu. 感謝🙏

First things first, I learned about Quadratic Kappa Metrix from [this note book](https://www.kaggle.com/code/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps/notebook). Thanks a lot.

# 1.EDA (Data Visualization) 
Here is the[ note book](https://www.kaggle.com/code/nhopet/pgss3e5-eda-heatmap-boxplot-outlierfilter).
I tried to visualize the data first in order to examine the data distribution. For me, it also serves as an exercise in drawing graphs. 

## 1)Heat Map 
I draw heat maps of train and test data in order to check feature correlations and  to compare  similarity in training and test data. I am not a scientist so I did not know what the correlation between the data meant. Although I found that the heatmaps of the train data and the test data show the same trend. This confirms that this data is valid to be treated for machine learning. 

## 2)Box Plot 
Boxplots were used to visualize the distribution of the data.  

### Outliers 
After plotting, we noticed a few outliers. We thought it would be better to exclude outliers in order to create a model with better accuracy.  I have posted [a discussion about handling way of outliers.](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/383544) I tried filtering outliers using the Isolation Fores based on [@Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis)' comment. I tried training on data with outliers removed. I spent a lot of time on it, but the scores were worse, so finally, I decided not to remove the outliers. 😭

###Quality Value Frequency 
From observation of the data, the data distribution of the train data and the test data seemed to be almost identical. This led me to believe that the score would be better if the frequency of quality values for the test data were predicted to match the frequency of quality values for the train data. The calculated code is this. 

#2.Model 
I have only tried xgboost. In previous playground series, I have tried various models. But at this time, instead of trying different models, I focused on finding a threshold that assigns the model's expected probability value to the quality value, which makes the quadratic weigthed kappa score optimal. 

##1)Simple xgboost
Here is [the notbook](https://www.kaggle.com/code/nhopet/pgss3e5-xgboost).
I first tried to simply choose the quality value with the largest predictive probability. The result of this model, public score was 0.49599. As expected, this was not so good. 
I also tested the train data with outliers removed and found a score of 0.44529. From this result, I determined that the removal of outliers did not have a positive impact on the model. 

##2)xgboost (weighting the model's predicted results)
Here is [the notebook](https://www.kaggle.com/code/nhopet/pgss3e5-xgboost-try2-weighted).
Next, I tried to weight the model's predicted results so that the distribution of predicted quality values for the test data would be similar to the distribution of quality values for the train data. To optimize the weighting, we used the minmize function of scipy optimize. Here is the code. As it turned out, this was a bad outcome. Forcing the predicted results to be assigned to 3, 4, and 8 made the rating values even worse. The score dropped to 0.39527.  This made me realize that I could not win this competition by forcing myself to predict good wines. 

##3)xgboost(optimize threshold)
Here is [the notebook](https://www.kaggle.com/code/nhopet/pgss5e3-xgboost-try3-optimizekappa).
The next step I tried was to convert the predicted value into an expected quality value, and then determine a certain threshold value for this expected value to be the quality value. The threshold for optimizing qudrautic weighted kappa was looked for using the minimize function in Spicy optimize. The code is this. This worked amazingly well.👌 The score went up to 0.544. 

The results of this projection did not include 3,4,8. I had my doubts. I am not able to predict good wine. So I posted [a discussion](https://www.kaggle.com/competitions/playground-series-s3e5/discussion/386270). 

#3.Hyperparameter Tuning 
I did it manually. I decided by moving some parameters manually.　 The score went up a bit to 0.55713. 

#4.Cross-Validation 
For cross-validation, I tried n_splits= 12, 5, 2 with StratifiedKFold. Hyperparameters were not tuned to the number of divisions. I did not do this because it was time consuming. 
Socres were n=12: 0.53736  n=5: 0.54088 n=2:0.54472 . Reducing n_splits seemed to increase the score. So I trained the model with all train data without CV. But this was scored down to 0.4819. 

As these result, the model trained by n_splits= 2 got 2nd place !! 🎉
Here is[ the 2nd place solution code](https://www.kaggle.com/code/nhopet/pgss5e3-2nd-code-xgboost-optimezekappa-cv2).
 
#5.Summary 
I think the winning factor was to find the threshold that optimizes the qudrautic weighted kappa. I came in second in the competition, but I was concerned about my inability to predict good taste wines.🙄 

Thanks again to all the competitors and to the organizers at Kaggle for another fun round in the Playground Series. 😄
