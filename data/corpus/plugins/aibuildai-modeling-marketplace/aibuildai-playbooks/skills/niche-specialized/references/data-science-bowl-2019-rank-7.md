# 7th Place Solution

Competition: data-science-bowl-2019
Rank: #7
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127213

# Approach
- The best part of the competition for me was feature engineering. In the end  I used 51 features, truncated down from 150.
- By far  strongest features were base distributions of each Assessment. However, they were used by everyone. Individual features, given small data, were not so important, but they decided winners I think.
- Final model was an ensemble: 0.3 LGB, 0.3 CATB, 0.4 NN
- 20 fold-bagging for all models, for NN additionally averaging across 3 seeds 
- One “trick” – use assessments from test set “blindly” (as we did not see this data) as samples in training. Data was scarce – so I was looking for every way to increase the number of training samples. Especially that you add data exactly for the children that are in the private LB. 

# Results
- Truncated CV: 0.575
- Private LB: 0.559
- Public LB: 0.559

# Final Remarks
- Congrats to winners - looking forward to your solutions
- Thanks to the organizers for the competition with event data – love those 😊 It unleashes your creativity in feature engineering

# Validation setup
- Truncate - select one assessment randomly for every child to reflect test set structure

# Update 1: Features
Motivation:
I was impacted by the ideas presented in this paper by Francois Chollet: [On the measure of intelligence](https://arxiv.org/abs/1911.01547). There are tons of interesting and powerful thoughts there. I was mostly stimulated by a discussion on how to measure intelligence: 
- A/ by overall-skill-level 
- B/ by skill-acquisition-tempo

In our case, we are measured by A/, which can be broken into two drivers:
- experience, i.e. how much time/effort the child has spent on various actitivities in the game. This formed my first group of features
- accuracy - how accurate was this child in her journey. This formed my second group of features.

However, skill-acquisition-tempo is a very interesting way to capture how quickly children are learning (features like minutes per level, events per level, etc.). This formed by 3rd group of features

I love competitions with manual feature engineering. Combination human+machine wins, which represents my view on how AI will impact the world.

#Update 2: Feature selection
- Calculated cv score after dropping a feature - did this individually for all ~150 features
- Dropped all features which brought an improvement of less than 0.0001 on QWK score - I treat them as noise. Found ~100 noise features in this way.
- Recalcuated  CV once more to see that overall score improved slightly after removing 100 noise features
