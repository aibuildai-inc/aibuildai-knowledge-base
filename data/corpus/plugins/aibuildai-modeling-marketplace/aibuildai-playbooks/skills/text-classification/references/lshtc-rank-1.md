# Winning Solution Description

Competition: lshtc
Rank: #1
Source: https://www.kaggle.com/c/lshtc/discussion/7980

<p>Hi,</p>
<p>I hope everyone had a good competition. Our team won after a close fight between the top 3 contenders. We've written up a description of our models and the code that can be used to reproduce the winning solution. In brief:</p>
<p>Our winning submission to the 2014 Kaggle competition for Large Scale Hierarchical Text Classification (LSHTC) consists mostly of an ensemble of sparse generative models extending Multinomial Naive Bayes. The base-classifiers consist of hierarchically smoothed models combining document, label, and hierarchy level Multinomials, with feature pre-processing using variants of TF-IDF and BM25. Additional diversification is introduced by different types of folds and random search optimization for different measures. The ensemble algorithm optimizes macroFscore by predicting the documents for each label, instead of the usual prediction of labels per document. Scores for documents are predicted by weighted voting of base-classifier outputs with a variant of Feature-Weighted Linear Stacking. The number of documents per label is chosen using label priors and thresholding of vote scores.</p>
<p>The full description .pdf file is attached, and the code can be downloaded from:&nbsp;https://storage.googleapis.com/kaggle-competitions/kaggle/3634/media/LSHTC4_winner_solution.zip</p>
<p>The above code package includes precomputed result files for the base-classifiers used by our ensemble. These take close to 300MB. A package omitting the base-classifier output files is also available: https://storage.googleapis.com/kaggle-competitions/kaggle/3634/media/LSHTC4_winner_solution_omit_resultsfiles.zip</p>
<p>Feel free to ask any questions about our solution.</p>
<p>Cheers,</p>
<p>-Antti</p>
