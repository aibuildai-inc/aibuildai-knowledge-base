# Congratulations everyone!

Competition: how-much-did-it-rain-ii
Rank: #11
Source: https://www.kaggle.com/c/how-much-did-it-rain-ii/discussion/17766#100517

If anyone cares, I used xgboost with the features:
1) TF-IDF -> SVD
2) An outliers and non-outliers partition of the data based upon Mark Landry's code (public scripts). Thanks Mark! The outliers are used as OOB data to predict onto the non-outliers train data and test data. These become meta features.
3) NA based summary statistics. So things that are NA are 1 and not NA are 0. Then do typical cross correlations, summary statistics like median, mean, sd, etc.
4) Cross Pearson correlations between columns using only pairwise complete cases.  
5) Word count type features.
6) First and Last time wise of the full feature vector.
I will post the full code and summary later, but that's the basic idea.  Then I bagged 10 XG boost models using different random seeds. I wanted to ensemble more, but I didn't really have enough time and XG seemed best for me. I'd love to get into ANNs more soon as I see the winner probably has some cool new ideas using NNs.

 **- EDIT: attached my code. It's based upon public scripts of Rain II and Crowdflower.**

![enter image description here][1]


  [1]: https://www.kaggle.com/blobs/download/forum-message-attachment-files/3400/rain2.pptx(2).svg
