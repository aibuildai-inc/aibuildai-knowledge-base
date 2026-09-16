# 9th place solution summary

Competition: cafa-5-protein-function-prediction
Rank: #9
Source: https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/462605

Thanks to the organizers and Kaggle for an interesting competition, along with a major thanks to the community for many helpful ideas and notebooks throughout  the submission period. Also thanks to my amazing teammates @aypyaypy and @ahmedelfazouan.

Below a short summary of our approach. The approach is an ensemble of two different ideas, that turned out to complement eachother well after merging.

# Feed-forward NN

1.  As input features, we went with: Prot-T5 embeddings, ESM2 embeddings, Taxo features and the (arguably more risky) QuickGO annotations.
2. We predict the most common x_amount of annnotations as a target after propagation. In the final solutions, we ensemble models which predict different amounts of most common annotations, varying from 2000 to 3500 targets with steps of 500. This turned out to result in an important leap in terms of public score.
3. **Modeling**: skip layers significantly increased CV and public score. In particular, each dense layer output is forwarded as input to a last layer, which receives a concatenated set of these layers.
4. Basic 10-fold cv scheme with seed ensembling for a slight score boost.

Our best single model reached 0.605.

Below a representation of the shift by using different targets + seed ensembling compared to single model. The mixed sub predicts pred=1 more confidently.



## Propagation

The second part consists of a less ML-based  approach. 

1. Merge SprofGO and QuickGO using the method introduced by kirill’s public NB (i.e. after taking the average, select the top45 of each aspect; https://www.kaggle.com/code/kirilldubovik/cafa5-tuning-merge-datasets).

2. Propagation process. If there is a gene ontology among the children whose probability is expected to be larger than its own, the probability of the parent GO is replaced by the maximum probability of the child GO. This operation is performed in such a way that it propagates from the leaf GO to the root.

A visualization of the propagation process result is visualized below. As may be expected, a large amount of probabilities are set to 1. The individual public score of this approach reaches 0.597.



### Extra

We have also tried  some other approaches, but sadly can not report on their performance due to the private score being calculated for the selected submissions only, which is understandable from a computational perspective.

Some of the stuff we tried:

- Correct for the overlap in train/test set by including a nested CV with pseudo labels
- Replace test predictions in the rows that already exist in the training test set by using the OOF values
- CNNs (did not perform well at all)
- Any other sort of embeddings reduced model performance
