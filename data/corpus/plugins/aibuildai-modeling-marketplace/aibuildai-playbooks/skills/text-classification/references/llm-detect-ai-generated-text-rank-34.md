# Winning Solution 34th Solution (Possible 22nd Solution) LB Public: 0.954 LB Private: 0.92

Competition: llm-detect-ai-generated-text
Rank: #34
Source: https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470353

**This project was made as an extension to the ML@Berkeley NMEP (New Member Education Program).**
**Please check out our organization if interested!**
https://ml.berkeley.edu
https://www.linkedin.com/company/machine-learning-at-berkeley/mycompany/

Private: 0.92 (Private Best: 0.929) Public: 0.954

Here are some approaches that allowed us to achieve 0.92 for private score, ending up at 34th. A similar approach was used in a different notebook to achieve 0.929 (unfortunately this was not selected for the final submission)

Referenced and learned a lot from:
[TF-IDF Model (from public notebook):](https://www.kaggle.com/code/batprem/llm-daigt-excluded-prompts)
Source: LLM DAIGT excluded prompts
Scores: Private - 0.895, Public - 0.963

1. **Ensemble Random Forest**
Added a random forest model for the ensemble to increase model diversity (used light weight that is similar to mnb)
```python
 ensemble = VotingClassifier(estimators=[('mnb',clf),
                                            ('sgd', sgd_model),
                                            ('lgb',lgb), 
                                            ('cat', cat),
                                            ('rf', rf_model)
                                           ],
                                weights=weights, voting='soft', n_jobs=-1)
```

2. **Balance the Distribution and Add Dataset Diversity**
https://github.com/panagiotisanagnostou/AI-GA#
Made a well balanced and diverse training data by adding additional train data. I checked the distribution of labels, and added additional dataset to make a well-balanced training data. Specifically, AI-GA (AI-Generated Abstracts dataset) was used. 

```python
merged_df['label'].hist(bins=10)  
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution')
plt.show()
```


Check that the distribution is now balanced after adding data


Please feel free to ask any follow up questions.
