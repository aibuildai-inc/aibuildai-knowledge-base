# 3rd Place Solution

Competition: march-machine-learning-mania-2024
Rank: #3
Source: https://www.kaggle.com/c/march-machine-learning-mania-2024/discussion/495101

**March Machine Learning Mania 2024**

[Notebook](https://www.kaggle.com/code/adambachmeier/silver-goto-conversion-predictions)

**Private Leaderboard Score:** 0.05536
**Private Leaderboard Place:** 3rd

**Background:**
I am a 2014 graduate of St. John's University with a degree in Computer Science. I have a decade of experience as a Software Engineer experience working on SaaS products but I have only recently begun working more closely within the Data Science sphere. In February, I participated in a three-week AI Bootcamp hosted by my employer’s Chief Data Scientist. The bootcamp was a crash course in data science, machine learning, and AI training covering everything from general statistics to advanced clustering techniques. As part of that course, my colleagues and attempted making predictions for the [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) prediction competition. This is my first competition.

**Solution:**
This approach utilizes Nate Silver's 2024 odds prediction matrix for both the Men's and Women's March Madness Tournaments. It simulates each matchup by converting the odds for two competing teams into probabilities that Team A or Team B will win, employing the goto_conversion method to adjust for favorite-longshot bias. These probabilities are then used to orchestrate the matchup through a weighted coin flip advancing the winner to the next position in the bracket.

**Execution:**
Data adjustments: Cleaning only
Number of brackets simulated: 99
Submission Entries: 1
Runtime: < 30 seconds

**Honorable Mentions:**
I'm realizing just how awesome this Kaggle community is and want to give special mention to those who indirectly helped with this submission.

- Thank you to Competition Host @jeffsonas for putting together this [thread](https://www.kaggle.com/competitions/march-machine-learning-mania-2024/discussion/484453) about how to submit if you did your work outside the notebook. My preferred working environment for Python is Visual Studio Code which is where I did most of my work.
- Thank you to @ryanchew1 for this posting this public [model](https://www.kaggle.com/code/ryanchew1/getting-started-with-madness) which was an excellent starting point for generating a sample submission.
- Most importantly, a very deserved recognition and thank you to @kaito510 for this [notebook](https://www.kaggle.com/code/kaito510/1xgold-2xsilvers-key-ingredient/notebook) from last year's competition, an updated [version](https://www.kaggle.com/code/kaito510/updated-1xgold-2xsilvers-key-ingredient) for this year's competition, and for authoring [goto_conversion](https://github.com/gotoConversion/goto_conversion) which was the "key ingredient" for this solution.
