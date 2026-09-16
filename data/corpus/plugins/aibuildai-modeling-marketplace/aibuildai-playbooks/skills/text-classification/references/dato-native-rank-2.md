# Congratulations to winners!

Competition: dato-native
Rank: #2
Source: https://www.kaggle.com/c/dato-native/discussion/17009#96206

Mad Professors: Congratulations, and good job!  Don't worry, I knew our positions could easily reverse.  It was fun to have a real shot at the top position for a while.

My submission was just the scaled sum of two models, a linear SVM of skip-grams and element/attribute combinations (79 million unique features, ~45 minutes training time), and an XGBoost model using substrings as features (30,000 features, 2000 trees, ~5 hours training time).

The substrings were automatically selected using this tool: https://github.com/mortehu/substring-frequencies

These are the substrings used as features in the XGBoost model, with the most predictive being listed first: https://gist.github.com/mortehu/07f0c59dc30bf495853d
