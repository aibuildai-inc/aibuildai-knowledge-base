# #18: Py-boost predicting t-scores

Competition: open-problems-single-cell-perturbations
Rank: #18
Source: https://www.kaggle.com/c/open-problems-single-cell-perturbations/discussion/458661

Did you notice that in this competition ~~no real EDA notebook has~~ few real EDA notebooks have been published? Besides explaining my machine learning model, I'd like to share some observations which help understand the data and the intricacies of Limma.

# Integration of biological knowledge

## Don't trust the cell types!

Let's recapitulate the course of the experiment in a simplified form. We can imagine an experimenter who is in front of a large pot of human blood cells. The pot contains a mixture of six cell types in certain proportions. T cells CD4+ take the largest share (42 %), only 2 % are T regulatory cells:


The experimenter now takes 145 droplets out of the large pot. Every droplet contains 1550 ± 240 cells (normally distributed). If we counted the cells per cell type in the droplets, we'd see a multinomial distribution. The 145 droplets might be composed like in the following bar chart (fictitious data, sorted from smallest to largest droplet):


In the next step, the experimenter adds 145 substances to the 145 droplets and waits 24 hours. After 24 hours the cells are analyzed. If we count the cells again, we get the following picture, as taken from the competition's training data (cell counts for the test data are hidden):


In this diagram we first see that some compounds are so toxic that in some droplets less than 100 cells survive. These droplets are represented by the leftmost bars in the bar chart.

The second observation is much more important: The long red part in the bars for Oprozomib and IN1451 show that these droplets contain several hundred T regulatory cells — much more than at the start of the experiment. Other compounds (e.g., CGM-079) have too many T cells CD8+ (green bar). How can we interpret this observation?
1. Does IN1451 incite the T regulatory cells to multiply so that we have five times more of them after 24 hours? No.
1. Does IN1451 magically convert NK cells into T regulatory cells? No.
1. Does IN1451 affect the cells in such a way that they are misclassified? Maybe.

Discussing differential gene expression for specific cell types becomes pointless if the cells change their type during the experiment. For the Kaggle competition this means that we have to deal with many outliers: Beyond the at least five toxic compounds, there are at least seven compounds which change the cells' types. Differential expression for these outliers is hard to model. They make cross-validation unreliable, and the outliers in the private leaderboard can't even be predicted by probing the public leaderboard.

## Cell count shouldn't affect differential gene expression

Does gene expression in a cell depend on how many cells are in the experiment? Theoretically, it doesn't. A cell behaves the same way whether there are 10 cells in the experiment or 10000. We'd expect, however, a difference in the significance of the experimental results: An experiment with 10000 cells should give more precise measurements than a 10-cell experiment: As the cell count grows, variance of the measurements should decrease, t-score should be farther away from zero, and pvalues should decrease.

The competition data don't fulfill this expectation. If we plot the mean t-scores versus the cell count for the 602 cell type–compound combinations (excluding the control compounds), we see a linear relationship: For every cell type, compounds with lower cell counts have positive t-score means, and compounds with higher cell counts have negative t-score means. This correlation between cell counts and t-scores shouldn't exist. It is an artefact of Limma rather than a biological effect.

You can plot the diagram with median or variance instead of mean — it will look similar. You can even compare the cell counts to the first principal component of the t-scores and see the same correlation. 



We can now put together a list of 20 compounds which are to be considered outliers because of low cell counts. Notice that we don't declare single rows of the dataset to be outliers, but all 86 rows related to the 20 compounds:

```
Outliers
--------
AT13387                           only 7 T regulatory cells
Alvocidib                         ≤10 for several cell types
BAY 61-3606                       mean t-score for CD8+ cells > 2
BMS-387032                        only 10 T cells CD8+, no Myeloid cells
Belinostat                        control compound with too many cells
CEP-18770 (Delanzomib)            ≤10 for several cell types
CGM-097                           too many T cells CD8+
CGP 60474                         ≤10 for several cell types
Dabrafenib                        control compound with too many cells
Ganetespib (STA-9090)             only 4 T regulatory cells, too many NK cells
I-BET151                          too many T cells CD8+
IN1451                            ≤10 for several cell types
LY2090314                         only 6 T cells CD8+
MLN 2238                          ≤10 for several cell types
Oprozomib (ONX 0912)              ≤10 for several cell types
Proscillaridin A;Proscillaridin-A ≤10 for several cell types
Resminostat                       no T cells CD8+
Scriptaid                         only 2 T regulatory cells
UNII-BXU45ZH6LI                   only 6 T cells CD8+
Vorinostat                        only 1 T regulatory cell
```

After removing the outliers, the diagram looks much cleaner. The variance of the cell counts remains. It is a source of noise which impedes the correct interpretation (and prediction) of differential expressions. Maybe we'd get cleaner data if we equalized the cell counts before library size normalization. This would amount to throwing away a part of the measurements, which isn't desirable either.



After considering the small size of the dataset, the amount of noise and the Limma artefacts (more of them will be shown in the next section), I didn't try to integrate any external biological data into my model. 

# Exploration of the problem

## A mixture of probability distributions

A histogram of a single row of the training data (18211 t-scores for T cells CD8+ treated with Scriptaid) shows that the distribution is multimodal.

The highest mode consists of 269 genes which all have an identical t-score of -3.769. It turns out that these are the 269 genes which are never expressed in T cells CD8+, neither with the negative control nor with any other compound. Isn't this strange? A gene which is never expressed in the whole experiment should have a log-fold change of zero and should not get a t-score at all (because t-score computation involves a division by the variance, and the variance of a never-expressed gene is zero).

For Myeloid cells treated with Foretinib, 3856 genes are not expressed (RNA count of zero), yet most of them have a positive t-score. Their highest t-score is 6.228 (resulting in a pvalue of 4e-10 and a log10pvalue of 9.33). If an RNA count is zero, the corresponding log-fold-change (and t-score) should never be positive.

We may say that the distribution of the values is a mixture of two distributions:
1. The values for the genes which are expressed (blue) have a more or less bell-shaped distribution.
2. The values for the genes which are not expressed (orange) have a distribution with an unusual shape, and it is strange that positive differential expressions are reported when not a single piece of RNA is counted.

What we see here is an artefact of Limma, which affects every row of the datset. It suggests that Limma output can be biased and is not ideal for investigating cell-type translation of differential expressions.

```
Genes expressed in T cells CD8+ Scriptaid:     13560
Genes not expressed in T cells CD8+ Scriptaid:  4651
Mode: -3.804 for 269 genes not expressed at all in T cells CD8+
```


```
Genes expressed in Myeloid cells Foretinib:     14355
Genes not expressed in Myeloid cells Foretinib:  3856
Mode: 6.379 for 81 genes not expressed at all in Myeloid cells
```

## An ideal training set

In the competition overview, the organizers ask: *Do you have any evidence to suggest how you might develop an ideal training set for cell type translation beyond random sampling of compounds in cell types? What is the relationship between the number of compounds measured in the held-out cell types and model performance?*

I think we are not yet ready to answer these questions. We first need cleaner data (and more of it):
- Cell types must be classified correctly. This may imply that we limit the scope of the work to compounds which do not hamper cell type classification.
- Samples containing too few cells must be eliminated from the dataset. These samples just add hay to the haystack where we want to find the needle.
- Even if we have many cells, genes with low rna counts may need to be eliminated. Otherwise they add even more hay to the haystack.


Second, modeling strange t-scores of genes which are never expressed is a waste of time. We need to define a machine-learning task and a metric which reward biological insight rather than forcing people into modeling the noise created by upstream processing steps:
- As t-scores are always affected by cell counts and variance estimates, a metric based on less highly-processed data (i.e., log-fold changes or rna counts rather than log10pvalues or t-scores) may lead research into a better direction.
- Even with log-fold changes, genes with low rna count make more noise than genes with high rna count. A suitable metric should account for this fact.

# Model design

## T-scores are better than log10pvalues

Limma performs t-tests. t-scores are (almost) normally distributed, which is good for machine learning inputs. For this competition, the t-scores were nonlinearly transformed to log10pvalues. The transformation squeezes the nice bell shape into a distribution with a much higher kurtosis.

My machine learning models perform better if I transform the log10pvalues into t-score in a preprocessing step, predict t-scores, and transform the predictions back afterwards. Perhaps working with log-fold changes or RNA counts would be even better.



## The models

I developed four models:
- Py-boost
- A recommender system based on ridge regression
- A recommender system based on k nearest neighbors
- ExtraTrees

I first implemented the Py-boost model, derived from @alexandervc's public notebook.

I then implemented the ExtraTrees model, which resembles @alexandervc's Py-boost model. All the decision trees are fully grown (i.e., overfitted). The model gets its generalization capability from noise which is added to the target-encoded features deliberately.

I then implemented the knn [recommender system](https://en.wikipedia.org/wiki/Recommender_system) to have some diversity in the ensemble. Cell types and compounds are identified with users and items, respectively; gene expression is identified with item ratings by users.

ExtraTrees and k-nearest-neighbors share the weakness that they cannot extrapolate. Even after dimensionality reduction, our training dataset essentially consists of 614 points in a high-dimensional space, so that most of the points will lie on the convex hull. Of the 255 test points, many will lie outside the convex hull of the training points, which means that the model must extrapolate. To bring the extrapolation capability into the game, I implemented the ridge regression model. 

The models have cv scores between 0.878 (ExtraTrees) and 0.906 (Py-boost). Py-boost, which was the worst in cross-validation, has the best public and private lb scores (0.572 and 0.748, respectively).

## Data augmentation

One of the models (k nearest neighbors) is fed with **data augmentation**: If we know the differential expressions for two compounds, we may assume that a mixture of the two compounds will produce a differential expression which is the average of the two single-compound differential expressions.

I experimented with another kind of data augmentation a well: Because there are more than twice as many T cells CD4+ as either Myeloid or B cells and I knew that the cell count biases the results of Limma, I reduced the cell count of the T cells CD4+, pseudobulked them, ran them through Limma and added the results to the training data as another cell type. This augmentation improved the scores of ExtraTrees, but not to the level of Py-boost. Perhaps I should have combined the additional cell type with Py-boost...

# Robustness

The robustness of my models is demonstrated in two ways:

(1) The models are fully cross-validated. The cross-validation strategy, first documented in [SCP Quickstart](https://www.kaggle.com/code/ambrosm/scp-quickstart), ensures that the model is validated on predicting cell_type–sm_name combinations so that it knows only 17 other compounds for the same cell type. This cross-validation strategy is more robust than the ordinary shuffled KFold, where the model knows 4/5 of all compounds for the same cell type. (And it is much more robust than a simple train-test-split.)

I have to admit, though, that I'm not happy with the cv–lb correspondence.



(2) For all models the performance was tested after adding Gaussian noise to the input t-scores. All models are robust against small noise. When the noise gets stronger, the knn and ExtraTrees models suffer more than Py-boost and the ridge recommender system.
 


# Documentation and code style

The code is documented in the notebooks.

# Reproducibility

Source code is here:
- [EDA which makes sense ⭐️⭐️⭐️⭐️⭐️](https://www.kaggle.com/code/ambrosm/scp-eda-which-makes-sense)
- [SCP #26: Py-boost, recommender system and ET](https://www.kaggle.com/code/ambrosm/scp-26-py-boost-recommender-system-and-et)
- [GitHub](https://github.com/Ambros-M/Single-Cell-Perturbations-2023)

# Conclusion

Let me conclude by summarizing the four main messages of this post:

1. Recommender systems are a promising starting point for developing models for cross-cell-type differential gene expression prediction. Because of commercial interests, recommender systems are a well-researched topic, and a lot of information is available.
2. Data augmentation is useful, and mixtures of compounds are a natural approach to data augmentation.
3. Although Kaggle competitions with data cleaning, outlier removal and unusual metrics are entertaining, the research objective would profit from another setting. Providing clean data and scoring with a well-understood metric would help participants focus on the real topic rather than the noise in the data.
4. We have seen that Limma in certain situations produces biased outputs. I hope that professional Limma users are aware of these effects and account for them when interpreting results in their research.
