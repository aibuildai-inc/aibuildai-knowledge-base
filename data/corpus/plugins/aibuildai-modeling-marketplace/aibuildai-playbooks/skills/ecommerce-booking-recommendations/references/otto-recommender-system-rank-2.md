# 2nd Place Solution(ONODERA part)

Competition: otto-recommender-system
Rank: #2
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382790

First of all, thank you for launching and organizing this terrific competition @pnormann.

I wanted to be first, but I don't really care so far.
I'd like to explain my part.

### Candidates
When I teamed up with @psilogram, he already has great candidates compared to mine.
So I decided to use his candidates.

### Features
#### Item2item Features
Also @psilogram already has splendid features, but there is room for improvement regarding CF features.
So I focused on item2item features and that consists of
- count
- time difference
- sequence difference(invented by @psilogram)
- 2 kind of weighted above features
- Aggregation of above
In total, I got 93 features. After this, I could generate almost 5k features using different combination(e.g. click to order, cart to order, etc...)
I use just 400~500 features eventually.


#### 1st stage prediction Features


#### Pseudo Event Features



### Models
I used XGBoost and CatBoost.


### Pipeline
After that 2nd stage, we blended our result ( @senkin13, @h4211819 ) by rank.

[my teammate's solution](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)



### Acknowledgments
If I hadn't used cuDF and cuML, I couldn't manage a lot of experiments.
Thanks [RAPIDS](https://rapids.ai/index.html)

