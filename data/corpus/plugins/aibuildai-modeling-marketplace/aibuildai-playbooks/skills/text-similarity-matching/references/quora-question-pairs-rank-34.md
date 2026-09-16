# Congrats to winners! Sharing my interesting discoveries and tricks

Competition: quora-question-pairs
Rank: #34
Source: https://www.kaggle.com/c/quora-question-pairs/discussion/34292

Congrats to all winners. This is my first taken serious participation. Although there are some leaks in this competition, I still learn a lot from it and my overall skills have been improved a lot.

In this competition, our team made two stages stacking with around 200 features and about 50 nn-features which come from LSTM trained with different inputs and different architectures. We  trained xgb, lightgbm and NN on these features and then do an ensemble selections.

The best single model without any reweight get a 0.176 log loss in cv by lightgbm. 

Instead of using a 0.375 re-weight function, we adjust the prediction probability by the intersection count of neighbors. Since we find that the distribution of leak data in training set and testing set are almost the same. So we make a guess that the validation score in leak data may reflect the leak data in LB. So we fixed the prediction with leak data and adjust the prediction of questions share no neighbors and we succeed.

The distribution is calculated by 

（ #intersection neighbour &gt;= i） / （ #intersection neighbour &gt;= 1）

| neighbor &gt; i | training set | testing set |

|--------------|--------------|-------------|

| 1            | 1.00         | 1.00        |

| 2            | 0.68         | 0.61        |

| 3            | 0.53         | 0.51        |

| 4            | 0.45         | 0.44        |

| 5            | 0.38         | 0.38        |

| 6            | 0.34         | 0.34        |

| 7            | 0.30         | 0.31        |

| 8            | 0.27         | 0.28        |

| 9            | 0.25         | 0.26        |

| 10           | 0.22         | 0.24        |

| 11           | 0.21         | 0.22        |

| 12           | 0.19         | 0.20        |

| 13           | 0.17         | 0.18        |

| 14           | 0.15         | 0.17        |

| 15           | 0.14         | 0.16        |

| 16           | 0.13         | 0.15        |

| 17           | 0.12         | 0.14        |

| 18           | 0.11         | 0.13        |

| 19           | 0.10         | 0.12        |


 And we  adjust the prediction of no intersection neighbour by 

a = 0.06 / 0.185 b = (1 - 0.06) / (1 - 0.185)

f(x) = ax / (ax + b(1-x))

And we gain 0.02 in LB when we find this tricks.  

However, in the last several days, we stuck a lot in ensemble and stacking. We suffer a lot from overfitting. So we dive into exploring the distribution of features in training set and testing set agian. First, we find that when ensemble LSTM that trained without leak features, the prediction in validation set with a lot neighbors  differ a lot from the testing set. For example, when the number of intersection neighbors  is more than 5, the average prediction is about 0.998 but in the testing set, it's just far more less like 0.85.  When we found this we re-trained our base LSTM features with leak features. And we again gain 0.03 in the LB from 0.134 to 0.131. And we stuck in overfitting again.  And we don't have any time to find the reason for that.
