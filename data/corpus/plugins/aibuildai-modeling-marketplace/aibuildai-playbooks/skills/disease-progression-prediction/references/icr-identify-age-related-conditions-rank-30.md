# 30th Place Solution for the ICR - Identifying Age-Related Conditions Competition

Competition: icr-identify-age-related-conditions
Rank: #30
Source: https://www.kaggle.com/c/icr-identify-age-related-conditions/discussion/432084

On the competition's closing day, I was surprised to see a message from my friend and, upon checking the leaderboard, I found that I had come in 30th place.

After submitting a few times early in the competition, I participated in HuBMAP, so I didn't expect to win a medal.

### **Context**

- Business context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview
- Data context: https://www.kaggle.com/competitions/icr-identify-age-related-conditions/data

## Solution

I haven't done anything special.

I used this excellent notebook: https://www.kaggle.com/code/datafan07/icr-simple-eda-baseline and I am grateful to [https://www.kaggle.com/datafan07](https://www.kaggle.com/datafan07).

The private score of this notebook was 0.37816, which is equivalent to a silver medal-worthy score.

I only made a modification in the stratified k-fold section of this notebook to perform a split that also considered EJ.

After reading this discussion (https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/411632), I realized that gender should also be considered during fold splitting.

```python
greeks = pd.merge(train[['Id', 'EJ']], greeks, on='Id')

for train_index,val_index in skf.split(train, greeks.iloc[:,1:-1]):
```
Due to this modification, the private score improved from 0.37816 to 0.37426.
