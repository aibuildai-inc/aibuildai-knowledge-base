# 2nd Place Solution for the Regression with an Abalone Dataset Competition

Competition: playground-series-s4e4
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s4e4/discussion/499698

Heyho everyone, thanks for hosting [the competition](https://www.kaggle.com/competitions/playground-series-s4e4), and shoutouts to the [active community](https://www.kaggle.com/competitions/playground-series-s4e4/discussion)! 

## Context for My Submissions

I entered the competition close to the deadline to test my automated machine learning (AutoML) workflow, which I intended to use for the [AutoML Grand Prix ](https://www.kaggle.com/automl-grand-prix) that started right after this competition ended. 

As it turns out, and to my surprise, my AutoML workflow performed really well without adjustments in this competition. In fact, it [worked much better here than on the actual first Grand Prix Dataset](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499482) because this dataset actually benefited from automated feature engineering. 

I will write up and share the code for the major parts of my workflow as part of my write-up for the Grand Prix. If possible, I will link my write-up here or post it as a comment later. **EDIT**: [here](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/500783) is the write-up. 

# Summary of the AutoML Workflow 

1. Automated Two-Sample Tests For Training Data
2. Automated Feature Engineering (OpenFE + AutoGluon)
3. Automated Model Selection (Customzied AutoGluon)

## 1. Automated Two-Sample Tests To Optimize Training Data

Since I am rather new to actively participating in Kaggle competitions, I found the concept of having an additional potentially unreliable data source quite interesting. Thus, to determine whether I wanted to include the original data in my training data, I created several two-sample tests for original data, training from the competition, and test data from the competition. These tests are supposed to determine if a distribution shift exists. If any shift exists, including the original data in the training data might be harmful.

I used AutoGluon to predict whether it can detect which dataset a sample is from and used a Mann-Whitney U Test to determine significance. 

I tested:
- (Kaggle+Original) Train Data - (Kaggle) Test Data
- (Kaggle) Train Data - (Original) Train Data
- (Kaggle) Train Data - (Kaggle) Test Data
- (Original) Train Data - (Kaggle) Test Data

Based on my observations from these tests, I included the original data. I still need to formalize a way of automatically deciding this. The setup can be improved, and I want to improve it in the future by including tests assessing whether including data is estimated to improve performance. 

## 2.  Automated Feature Engineering (OpenFE + AutoGluon)

I suspect this part was by far the most important in obtaining second place in this submission. In essence, what I did was: A) run a customized version of [OpenFE](https://github.com/IIIS-Li-Group/OpenFE), and then B) use [AutoGluon's feature selection](https://github.com/autogluon/autogluon/blob/master/core/src/autogluon/core/utils/feature_selection.py#L154) to prune features.

I played around with B) and used a LightGBM model as a proxy. 

For A), I first selected the potential feature candidates by hand to control what might be searched over by OpenFE and added a new type of feature generator. Then, I ran OpenFE with default settings, which got me to ca. 200 features. Finally, I automatically pruned them down using B) with a time limit of 1 hour. 

Sadly, I do not have all the details for the final feature generators, as my pipeline did not log their generator function. Moreover, while trying to make automated feature engineering work for the first AutoML Grand Prix dataset (spoiler: it was not worth the effort), my code base got a bit too messy to trace back my steps. 

I used the following code to control the feature generators: 

```python
all_features = list(train_data.columns)
soft_ordinal = [f for f in all_features if (train_data[f].nunique() <= 100) and (f not in ordinal_columns)]
numerical_features = [f for f in all_features if f not in categorical_columns]
candidate_features = get_candidate_features(
    numerical_features=numerical_features,
    categorical_features=categorical_columns,
    ordinal_features=ordinal_columns + soft_ordinal,
    order=1,  # 2 is likely impossible to use w/o time estimate.
)

# Restrict Search Space of Candidate Features
candidate_features = [
    f
    for f in candidate_features
    if f.name
    in [
        # "abs" -> dataset specific, not useful in most cases
        # "log" -> can be done by scalers, no need for GBDTs
        # "sqrt", -> see above (s.a.)
        # "square" , -> s.a.
        # "sigmoid" , -> s.a.
        "freq", -> # do not like it but giving it a shot. 
        "round",
        "residual",
        # "max", -> IMO, trivial to model for first-order features 
        # "min", -> s.a.
        # "+", -> s.a.
        # "-", -> s.a.
        "/",
        "*",
        # "GroupByThenMin",  ->  The essential benefit of GroupBy is captured with any of these, so I filtered this to reduce the search space.
        # "GroupByThenMax", -> s.a.
        # "GroupByThenMean", -> s.a.
        "GroupByThenMedian",
        "GroupByThenStd",
        # "GroupByThenRank", -> s.a.
        "GroupByThenFreq",
        "GroupByThenNUnique",
        "Combine",
        # New Generators 
        #   - Hacked into OpenFE by adding `new_data = int(d < d.quantile(X).max())` to the generator options. 
         "<p0.2",  # X = 0.2
	"<p0.4",   
	"<p0.6",
	"<p0.8",
    ]
]
```  

## 3. Automated Model Selection (Customzied AutoGluon)

Finally, I used a "tuned" version of AutoGluon to select the final model. I will share all the details in my write-up for the AutoML Grand Prix. **EDIT**: [here](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/500783) is the write-up. 

Of importance here was that I noticed that stacking worked really well for this dataset (with code related to [dynamic stacking](https://github.com/autogluon/autogluon/pull/3616)). Thus, I set AutoGluon to fit up to 6 Layers for Stacking. In the end, it used 5 Layers, but looking at the final scores of my submissions, I got the same score with only 3 layers, but layer 5 did not overfit either! 

I hope this (and the more detailed future write-up) provides a good overview of my solution. Moreover, I hope you are as excited for more and better automated feature engineering as I am ;) 


Best,
Lennart
