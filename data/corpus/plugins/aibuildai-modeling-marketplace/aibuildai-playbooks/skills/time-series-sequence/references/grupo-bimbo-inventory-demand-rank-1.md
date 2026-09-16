# #1 Place Solution of The Slippery Appraisals team

Competition: grupo-bimbo-inventory-demand
Rank: #1
Source: https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23863

**Summary**

Overall solution was 2nd-level ensembling. We built a lot of models on 1st level (using 9 week as validation set) The training method of most of 1st level models was XGBoost.  For second level we used ExtraTrees classifier and linear model from Python scikit-learn. The final result is weighted average of these two models.
The most important features are based on the 1-3 weeks lags of target variable grouped by factors and their combinations,  aggregated features (min, max, mean, sum) of target variable grouped by factors and their combinations, frequency features of factors variables.
For our work, we used R environment with Rstudio IDE, and Python with Jupyter. We combined our work on our personal computers with cloud computing on x1.32xlarge Amazon EC2 instance with web interface for access. Last week of the competition was the continuous calculation of single and 2nd  level models on x1.32xlarge Amazon EC2 instance.

It was a very interesting competition, our approaches were inspired by many discussions on the competition forum and by many kernels published by other participants.

**Features Selection / Engineering**

One of the main ideas in our approach is that it is important to know what were the previous weeks sales. If previous week too many products were supplied and they were not sold, next week this product amount, supplied to the same store, will be decreased. So, it is very important to include lagged values of target variable Demanda_uni_equil as a feature to predict next sales. 
The simplified version of the R script [“Bimbo XGBoost R script LB:0.457”][1] for such an approach was published on the Kaggle forum of this competition. 
 We merged transactions data frame with data frames from files cliente_tabla.csv, producto_tabla.csv using appropriate key fields. We used frequencies features grouped by categorical variables (e.g. Producto_ID, Cliente_ID, Agencia_ID, etc. ) and by different combinations of them. Data frame with  new product features received by text parsing of product name variable NombreProducto from file producto_tabla.csv was merged to main data frame via key Producto_ID. As a result of parsing of product names we got a lot of new product features which denote e.g. short names of product, weight, brand, etc. We grouped the products by clusters and used the number of clusters as a new feature. 
 We chose the data of previous weeks in respect of training sets weeks for aggregations of target variable Demanda_uni_equil, grouped by factors variables. We calculated mean, min, max, sum of target variable grouped by factors and their combinations. Then these aggregated values was merged into main data sets as the new features using the list of these factors as keys for merging. We also calculated the similar aggregated features grouped by factors for variables Venta_hoy, Venta_uni_hoy, Dev_uni_proxima, Dev_uni_proxima. 
We chose the set of 8-9 weeks for the validation and 6-7 weeks’ data were considered as a training set for validation. It was necessary to build 2-level models and find appropriate weights for models blending. 8-9th  weeks were chosen as a training set for prediction sales of the 10-11th  weeks. So, we generated two types of training the data set: 6-7th  weeks training data set for the prediction of 8-9th  weeks sales and 8-9th  weeks for the prediction of 10-11th weeks sales. Since we used the 1st week lag for the target variable as a feature,  we needed to calculate this feature additionally for predicting 11th week sales. First we predicted the10th week, then using the predicted target variable for the 10th week, we calculated lagged features with the lag for 1 week and used these calculated features for the prediction of 11th week sales. As the first step for validation, we calculated the prediction for the 8th week, then using predicted target values we calculated 1 week lag for 9th week data set, and then we made the prediction for 9th week sales. It was done to build the validation model similar to the prediction of 10-11 weeks sales. For lag values, we investigated two types of cases with lags for 1-3 weeks and lags for 2-3 weeks. First we used up to 5 weeks lags, on the next study we used maximum 3 weeks lags. In the case of 1-3 weeks lags, for the validation on 8-9th weeks sales, we made the prediction in 2 steps. For building a two-level model, we used only the validation of 9th  week sales. Our study shows that using 1-3 weeks lags gives us better scores comparing with 2-3 weeks lags. For the features with 1 week lags we used only the mean value of target variable grouped by the list of factor variables.  For 2-3 weeks lags, we also used averaged lags for such features as Dev_proxima, Dev_uni_proxima, Venta_hoy grouped by the list of different factors. Most of the lagged features which were used for classification have 2 weeks lag. We generated more than 300 features.  All received results for the validation on 9th week sales and for the prediction of 10-11th weeks, were used for building two-level models. Our study also shows that using only 7th  week data for the validation on 8-9th  weeks and 9th week for the prediction of sales on 10-11th weeks gives better scores comparing with two weeks training sets, so we built our next single models based on the sets of 7th week for validation and on the sets of 9th week for the prediction. To speed up our calculation we selected 175 top features using function xgb.importance() of ‘xgboost’ R package and then we worked with this set of features. We  also calculated the predictions for Venta_hoy", "Venta_uni_hoy", "Dev_uni_proxima", "Dev_proxima" variables which were used on the 2nd level model. On the first level each of us used his own classifier options and features sets which gave results with good scores for single models without high correlation that is important for creating 2 level models. 

**Training Methods**

For creating single models of the first level, we used xgboost classifier. The options for XGBoost classifiers can be found in our scripts. On the second level, we used validation results of the first level models as training sets for the prediction of target variable for the test set. We created a linear model and model based on ExtraTreesClassifier. On the third level, the results from the second level models were just weighted average.

**References**

https://www.kaggle.com/lyytinen/grupo-bimbo-inventory-demand/basic-preprocessing-for-products
https://www.kaggle.com/ybabakhin/grupo-bimbo-inventory-demand/products-clustering
https://www.kaggle.com/mlandry/grupo-bimbo-inventory-demand/h2o-gbm/run/263502


Team Slippery Appraisals
Year 2016


  [1]: https://www.kaggle.com/bpavlyshenko/grupo-bimbo-inventory-demand/bimbo-xgboost-r-script-lb-0-457/discussion
