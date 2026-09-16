# 2nd place solution

Competition: commonlitreadabilityprize
Rank: #2
Source: https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258328

First of all, I would like to thank Kaggle and host for hosting such an interesting competition! 

## Summary
I chose the final sub the one with the best Public and the one with the best CV.
|  | CV|Public|Private|
| --- | --- |--- |
 |best Public|0.4503 | 0.444 |0.446|
 |best CV|0.4449 | 0.447 |0.447|

The following is about best Public.(In best CV, the weights were determined by nelder-mead and the restriction of weight summing to 1 has been removed. And the models were partially changed from best Public. )
I ensembled 19 models and did a post process. I adjusted the weights of the model by looking at the LB and CV. I've included negative values for weights as well as positive ones. In the post process, I multiplied different coefficients depending on the predicted value.


## Cross Validatioin
I used the following.
https://www.kaggle.com/abhishek/step-1-create-folds


## Model & Weight
I trained the model with dropout set to 0 except for 1 and 2. And, I did mlm pretraining only for model 3. Weights were calculated by nelder-mead and then tuned for higher LB
| model | CV | Public | weight |
| --- | --- |
|1.  roberta-base -> svr | 0.500 |0.476|0.020|
|2. roberta-base -> ridge | 0.500 ||0.020|
|3. roberta-base| 0.485 |0.476|0.040|
|4. roberta-large| 0.483 |0.463|0.088|
|5. muppet-roberta-large| 0.480 |0.466|0.022|
|6.  bart-large | 0.476 |0.469|0.090|
|7.  electra-large | 0.483 |0.470|0.050|
|8.  funnel-large-base| 0.479 |0.471|0.050|
|9.  deberta-large| 0.481 |0.460|0.230|
|10.  deberta-v2-xlarge | 0.486 |0.466|0.050|
|11.  mpnet-base | 0.482 |0.470|0.130|
|12.  deberta-v2-xxlarge | 0.482 |0.465|0.140|
|13.  funnel-large | 0.475 |0.464|0.110|
|14.  gpt2-medium | 0.498 |0.478|0.170|
|15.  albert-v2-xxlarge | 0.486 |0.467|0.120|
|16.  electra-base | 0.493 ||-0.170|
|17.  bert-base-uncased |0.507  ||-0.140|
|18.  t5-large |0.504  ||-0.110|
|19.  distilbart-cnn-12-6 |0.489  |0.479 |0.090|

## Post process
Post process improved the score by about 0.001 ~ 0.002. The coefficients were calculated by nelder-mead and then tuned by looking at Public. The threshold was determined by looking at CV and Public
- pred >= 0.3  ->  pred * 1.07
- 0.3 > pred >= 0 ->  pred * 1.2
- 0 > pred >= -0.7 ->  pred * 0.974
- -0.7 > pred >= -0.9 ->  pred * 1.01
- -0.9 > pred > =-2 ->  pred * 1.021
- -2  > pred ->  pred * 1.027

## GPU
- A100(GCP)
- V100(GCP)
- P100(kaggle)

## Edit
Inference notebook(final sub)
https://www.kaggle.com/takoihiraokazu/lb-ensemble-add-electra-base-bert-base2-t5-diba2?scriptVersionId=69659799
Inference notebook(Partially modified version)
https://www.kaggle.com/takoihiraokazu/final-sub1
code
https://github.com/TakoiHirokazu/kaggle_commonLit_readability_prize
