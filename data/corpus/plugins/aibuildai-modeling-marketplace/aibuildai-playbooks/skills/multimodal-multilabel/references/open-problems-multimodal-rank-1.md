# 1st Place Solution Summary

Competition: open-problems-multimodal
Rank: #1
Source: https://www.kaggle.com/c/open-problems-multimodal/discussion/366961

First of all, thank you to the organizers and kaggle management and to everyone who participated with me.
Since I needed to gain experience in analyzing single cell data, this competition was an excellent experience for me.

I would like to introduce the overview of my solution.

# Multiome
## Model Overview


## Input Preprocessing


## Target Preprocessing


tSVD-based imputation method: 
1. Perform dimensionality reduction on the data with tSVD
2. And then, Transform the data back to the original space
3. Copy the value of the 0 part of the original data from the transformed values.

## Model


## Output Postprocessing and Loss




In the inference phase, the model outputs the average of the five predicted target data.

# CITEseq
## Model Overview



## Input Preprocessing


In selecting important genes in CITEseq, the correlation coefficient is calculated for each batch and select only genes with high correlation in many batches.
Genes were selected from those related to the target proteins and pathway.
I use [Reactome](https://reactome.org/) as pathway database.

## Target Preprocessing


## Model



## Output Postprocessing and Loss


In the inference phase, the model outputs the average of the five predicted target data.

# Local evaluation
I used two evaluation schemes.

1. Evaluation with cross validation:
  * 5-fold cross validation grouped by donor and day
2. Evaluation for hyperparameter optimization with Optuna:
  * Training data set is divided into training and validation data sets. ( Training data set: 80%, validation data set: 20%. )

# Ensemble
I used the weighted average of predictions of the following models.
1. Models trained with changing the seed 
2. Models fine-tuned on only some batches
  * Batch combination pattern examples: males only, female only, Day 4, 7 only, etc.
  * Use a model trained on the full training data set as a pre-training model 

# Code
https://github.com/shu65/open-problems-multimodal


# Update
2022/11/20 add the repository url of my solution
2022/11/26 fix some figures
