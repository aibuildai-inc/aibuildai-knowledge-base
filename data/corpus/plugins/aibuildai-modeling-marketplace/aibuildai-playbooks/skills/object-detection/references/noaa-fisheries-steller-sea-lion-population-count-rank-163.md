# Solution sharing

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #163
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35415

Congratulations to the winners and thanks to the organizers and fellow competitors. What I really enjoyed in this challenging competition was the fact that the Customer is a scientific agency with a good cause.

* The approaches I experimented with:
    1. Five binary linear SVMs (background vs sea lion type, one model for each class) that use HOG features and iteratively search and train-on hard examples. Result: Detecting too much background as sea lions.
    2. Six class CNN (five sea lion types + background) trained with and without Euclidean distances to 50 nearest sea lions (for each sea lion type). The idea with the distances was to train a NN to learn the group behavior of sea lions. Result: (i) With the distances: The problem here was that you only know the distances for the training data. In the testing phase, I replaced the distances by a missing value identifier, which apparently was the cause for the NN to output only background predictions. (ii) Without the distances: much better predictions than in the HOG solution, but still way too much background getting detected as sea lions.
    3. Binary CNN operating on 512x512 patches to first identify the spots in the images where it's probable to have sea lions, and then passing images of those spots to CNN regressor outputting counts for each sea lion type. Result: By manual inspection, these predictions looked actually pretty good, but the LB scores were only slightly better than the all-zeros submission.
    4. Extracting features from 512x512 patches through pre-trained VGG16 and feeding these features to five gradient boosted regression trees (one for each sea lion type). Result: My best performing solution, scoring 23.9 on private LB.


* An estimate of the time spent on this competition:
    * 2/6 programming and innovating data pre-processing.
    * 3/6 programming and innovating machine learning models.
    * 1/6 waiting until the computation finishes.
