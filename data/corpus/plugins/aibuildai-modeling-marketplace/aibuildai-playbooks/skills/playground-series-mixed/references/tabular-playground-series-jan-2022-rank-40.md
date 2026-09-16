# [40th place] Boltzmann Ensemble: A solution from statistical thermodynamics

Competition: tabular-playground-series-jan-2022
Rank: #40
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304353

This was a fun competition, characterised by an atmosphere of openness and only friendly rivalry. Thanks to all who have taken part, I hope you’ve all enjoyed it, learned more about data science, and that newer Kagglers have gained the confidence to compete in medal-bearing competitions in future.

For this month’s TPS, I chose to test out an ensembling method inspired by statistical thermodynamics. It has only a single optimisable parameter. Each model represented in the ensemble is weighted by a factor that depends in an exponentially decaying manner on its public LB score. Specifically, each model is given an exponential weight according to

exp(b*(S-x))

where x is the public LB score of that model. Larger scores are worse, hence the weights get smaller as x increases and get larger as x decreases. Thus, the best models have the largest weights and make the largest contributions to the ensemble.

The parameter b is the one meaningfully adjustable parameter of the model, the larger b is then the faster the weights decay as the score gets worse. The models in this ensemble can be thought of as corresponding to quantum states in statistical thermodynamics.

Evidently, b is analogous to 1/kT in statistical thermodynamics and behaves like a reciprocal temperature. In thermodynamics textbooks this parameter is often called beta. 

The LB score x is analogous to energy, since higher LB scores are bad in this competition the weights, which are effectively Boltzmann factors, decrease with increasing x. The opposite sign convention would have been used in a competition where higher LB scores are good. 
S is a calibration parameter defined such that if S is set to the best single model score then the highest weight is 1.0, which is convenient, but not essential.

The sum of the weights q is analogous to the partition function in statistical thermodynamics, and measures an effective number of contributing models. This is analogous to how the partition function, if calibrated such that the highest weight is 1.0, measures the number of thermally accessible quantum states.

For this TPS competition, many Kagglers shared excellent notebooks containing models scoring close to the top of the leaderboard. This would be unlikely in the more competitive environment of a higher stakes competition with medals on the line, so if you plan to use this or any other ensembling approach there then you will probably need to be able to include some fairly strong private models of your own. 

The strength of this method is that there is only one parameter to optimise, which means that there is no need to fit the weight of each model separately. However, I found that the best value of b changed significantly during the competition.

A strength can also be a weakness, and the approach has limited flexibility. The choice of which notebooks and versions to include, a model is either fully in or fully out, and in this presentation there is no account taken of similarity between models. A more sophisticated approach might penalise models which are strongly correlated with others already in the ensemble, however this would increase the number of parameters. Also, there is no flexibility here to include the sometimes useful possibility of negative weights. The Boltzmann Ensemble method works best when the target is a continuous numerical quantity. 

You can read the notebook [here](https://www.kaggle.com/jbomitchell/boltzmann-ensemble-without-overfitting)
