# [22nd place] From 325th in Public LB to 22nd in Private LB

Competition: petfinder-adoption-prediction
Rank: #22
Source: https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/89768#latest-519578

Hi Kagglers, this is my first competition and I am excited to share our solution. Please provide feedback if you spot anything!


We’d like to thank for Kaggle community and Petfinder.my for this learning experience. We learned a lot from those who share their knowledge and want to share our method with the hope of contributing to the community.

[See our kernel here](https://www.kaggle.com/enisimsar/22nd-place-solution)

# Data Loading and Feature Extraction

## Metadata, Sentiment, Image and Their Aggregations

- We took this part from [Single XGBoost model](https://www.kaggle.com/ranjoranjan/single-xgboost-model) 
- Addition to kernel above, we added min, max and std to its aggregations. Image number for each pet is also added.
## Breed Features
### Handling Mix Breed
We want to detect mixed breed instances and put specific(values other than 0-unknown- or 307-mixed-) breed information to *Breed1* column. Here is our strategy.
1. If any of *Breed1* or *Breed2* is mix breed(denoted by 307), we set *mixed_breed* column to 1. Then we change 307 value with 0.
2. For each mixed breed instance we set Breed1 as Breed1 + Breed2. Aim is to hold specific breed information in the *Breed1* column as much as possible.
3. If *Breed1* and *Breed2* columns are different than 0 for an instance, we set *mixed_breed* column to 1.

### Breed Prediction
For some instances we do not have specific breed type(*Breed1 == 0* and *mixed_breed=1*). We’ve applied breed prediction only for dogs since we were only be able to find proper dataset for dogs. 

[See our dog breed type prediction code here](https://github.com/sez-ai/Dog-Breed-Classifier)
[Get pretrained model here](https://www.kaggle.com/enisimsar/dog-identification-pretrained)

We insert a column *pred_breed*, and fill that column with predicted values. If we do not predict breed since that instance already has, we simply copy *Breed1* value to *pred_breed*.

### Creating pseudo-feature from breed names
Search breed names, if they contain any of the adjective below, add it as a categorical feature.
[‘hair’, ’domestic’, ‘short’, ‘medium’, ‘retriever’, ‘terrier’, ‘tabby’, ‘long’]

## Age
9 Bins are created for age column. See code for details.
## Name
Following features are extracted from names. [‘Name_len’,’Name_isna’,”Name_with_numbers”]
## Fee
5 Bins are created for fee column. See code for details.
## State
We used *state_gdp* and *state_population*  just like everyone else.
## Word - Sentence Embeddings
We used Spacy’s `en_vectors_web_lg` and applied SVD with n_components 32.
## Image Features
We were too lazy to use all images thus only used first photos of pets… Bottleneck image features are extracted from DenseNet121, Inception V3 and Xception with applying SVD for each one.
## External Data for Breed
- Scrapped from petfinder.com, note that it’s not petfinder.my. 
[petfinder.com External Data](https://www.kaggle.com/enisimsar/petfindercomexternal) 
- [Cat and dog breeds parameters](https://www.kaggle.com/hocop1/cat-and-dog-breeds-parameters)
- [Pet Breed Characteristics](https://www.kaggle.com/rturley/pet-breed-characteristics)
## Rescuer ID
There were many discussion about Rescuer ID, we also observed it causes overfitting. Thus one of our kernel was playing with rescuer ID while other was dropping this feature. Well, sometimes overfitting wins…

[See code for details](https://www.kaggle.com/enisimsar/final-submission-with-rescuer-features#Rescuer-ID)
## Aggregations
To enrich features even more, we have extracted many aggregations.
[See code for details](https://www.kaggle.com/enisimsar/final-submission-with-rescuer-features#Aggregations)
## Freq Encoder Categorical Features
Thanks to [Sample categorical feature encoding methods](https://www.kaggle.com/c/avito-demand-prediction/discussion/55521)
# Model
We used 2 models, XGBoost and LightGBM. Golden Section Search method is used for OptimizedRounder. Thanks to @hocop1 at [How to use regression here? | Kaggle](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107).
We’ve called OptimizedRounder in every fold and stored coefficients for each model. Then took mean of that coefficients for final use. 

As for final submission we’ve simply averaged coefficients and regression output of LGBM and XGBoost.
