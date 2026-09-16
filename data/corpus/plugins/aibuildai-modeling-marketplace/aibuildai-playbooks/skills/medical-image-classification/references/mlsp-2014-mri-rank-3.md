# 3rd position (solution)

Competition: mlsp-2014-mri
Rank: #3
Source: https://www.kaggle.com/c/mlsp-2014-mri/discussion/9967

Hello Everyone,

Sorry for the late posting. I was on holiday and could not access my main computer for a few weeks.

So my 3rd place entry is quite simple. I used what I think is a lesser known method called Distance Weighted Discriminant (DWD). This is a method designed to improve over LDA and SVM in high-dimensional low sample size setting. You can find a few quick explanations on how it achieves that in the pdf and further in the references cited therein.

DWD has one main parameter to tune: penalizing constant "C" (similar as in SVM case). I ran 10-fold cross validation with several values of C and found out that the performance (ROC area) saturated after around C=300.

The submission was done running out-of-the-box DWD on the data with C=300 and no feature selection. That was my winning entry. Which is a bit of a downer since I used this competition to test the performance of my LDA variant and one of the methods I compare it to is DWD.

Anyway I think it's a cool method and in my opinion too few people know about it. So hopefully it will reach larger audience and maybe someone will remember it if he/she will ever run into d&gt;&gt;n classification task in the future.

Here is the GIThub link: https://github.com/KarolisKoncevicius/Kaggle-MLSP-Schizo-3rd (I moved all the steps into a single R-script file).

The documentation is attached.
