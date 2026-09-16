# 21st place solution (link to R kernel)

Competition: data-science-bowl-2019
Rank: #21
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/127543

First at all, thanks to the BS Kids and Kaggle teams for this great competition and congratulations to the winners and medallists.

I couldn't work in this competition as much as how I would have liked so it's felt really good to get a medal. It's a bit disappointing to be so close to gold but I can't complain as my solution is very simple and I think I was lucky with the final result.


## Feature engineering
I generated 754 features, most of them very similar to the ones you can find in public kernels. For sessions of the type "Game" I created features taking account the different rounds (most games have three rounds).

## Feature elimination
I only drop duplicated and very similar (&gt;.99 equal values) variables. I ended with 649 features.

## Model    
I used the [1st place solution](https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19010) and the [2nd place solution](https://www.kaggle.com/c/prudential-life-insurance-assessment/discussion/19003) of the [Prudential Life competition](https://www.kaggle.com/c/prudential-life-insurance-assessment) as inspiration. My model consist of, first, three lgb binary classifiers (0 vs 123, 01 vs 23, 012 vs 3) with 5-Fold CV.  Then, I use the results of these models plus the assessment title as features of a linear regression model to get the final continuous prediction.

## Threshold definition
I use the `optim` R function with the Nelder-Mead algorithm. To get the initial coefficients I used the golden section method that is explained [here.](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/76107#480970) I usually got a better score with this two step process that using any of this two methods on their own. 

##.
You can see the kernel [here](https://www.kaggle.com/artmatician/21st-place-solution?scriptVersionId=27558325).
