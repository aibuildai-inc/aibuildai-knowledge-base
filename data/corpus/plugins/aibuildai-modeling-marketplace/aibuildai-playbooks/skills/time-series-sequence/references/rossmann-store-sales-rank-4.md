# Share your solution

Competition: rossmann-store-sales
Rank: #4
Source: https://www.kaggle.com/c/rossmann-store-sales/discussion/17896#101487

Hey Kagglers!    

Here’s a description of my solution (rank 36 public, 4 private) along with some **questions for the winners** at the bottom on their impressive results.

Overview and Set Up
-------------------

Like most of you I used an ensemble of xgboost models, a geometric weighted average of around six models with each having no more than 25 features.    The time-series cross-sectional (aka panel) structure of these data brings a Pandora’s box of good features, and I see now I only partially opened it.

For CV, I started out using the last six weeks as a holdout, but got stuck for a while and concerned about the high scores from the first week of July 2015 (especially July 4, a hot Saturday with low sales).   So I ended up switching to the final three weeks.   Local scores on it were typically close to LB scores, but I did encounter some models that were inconsistent.  The LB had me singing http://www.azlyrics.com/lyrics/eagles/witchywoman.html  at times.

For outliers, I ended up filtering out training observations with sales < 900.   The seven points with sales < 500 were troublesome but seemed to help some of my early models.

For features, I spent some time with basic feature engineering but more on feature selection, wielding Occam’s Razor hoping it did not turn into Sweeney Todd’s.  My best single model (0.09727 public, 0.10713 private, just glimpsing Gertland) has only 22 features:

    c("WeekOfMonth","month","week","day","Store","Promo","DayOfWeek","year","SchoolHoliday","CompDist0","CompOpenSince0","Promo2Since0","MeanLogSalesByStore","MeanLogSalesByState","MeanLogSalesByStateHoliday","MeanLogSalesByAssortment","MeanLogSalesByPromoInterval","MeanLogSalesByStorePromoDOW","MeanLogCustByStorePromoDOW","MeanLogSalesBySchoolHoliday2Type","Max_TemperatureC","SONNENSCHEINDAUER")

The last one came from DWD and gave me a LOL:  if your model is feeling gloomy just add a little sunshine!   To be safe I computed the mean features without using the holdout data while doing CV then switched to the full training data when building a submission.   The last mean feature was constructed using types of school holidays from the link kindly provided by Tobias Wolfanger (he and I must be distant cousins ;).  The simplicity of the models likely helped them to generalize, especially with the nadir of sales occurring near the public-private split point.

XGBoost Details
---------------

I modeled sales on the log scale and used a custom objective function:

    RMSPE_Objective2 <- function(preds, dtrain) {
      labels <- getinfo(dtrain, "label")
      elabs <- exp(as.numeric(labels))
      epreds <- exp(as.numeric(preds))
      eratio <- epreds/elabs
      hess <- eratio**2
      grad <- hess - eratio
      return(list(grad = grad, hess = hess))
    }

Note the Hessian here is not correct as it ignores the chain rule; it is flatter than the correct one and seemed to perform better. 
 
I did a fair bit of tuning and settled in on the following parameters:

    param <- list(objective = RMSPE_Objective2, eta = 0.01, max_depth = 11, 
                  subsample = 0.9, colsample_bytree = 0.3, min_child_weight = 20)

with 20K rounds and early stop 900.  Each run took 6-7 hours on some old Windows servers and to make things more difficult there was some instability in results so I ended up averaging results from 3 different seeds.   At one point I had 36 concurrent R sessions running :). 

To estimate linear adjustment factors I used

    dh$Sales1 <- dh$Sales/1000
    fit <- lm(Sales1~Pred,data=dh,weights=1/Sales1^2)
    print(summary(fit))

on the holdout results, took the intercept and slope estimates, and applied them to the corresponding LB predictions.  Typical values were -0.05 and 0.995 respectively.

I made simple ensembles as time moved along using the best performing models.

Things that Did Not Work So Well
--------------------------------

- Transformations other than log (original scale with weights, reciprocal, shifted log, Johnson Su)
- Google trends
- Correct Hessian and lamba = 0.13
- Mapping stores within a state
- Detrending

Software
--------

- JMP: data exploration, graphics, quick interactive analyses,
   submission checking and ensembling 
- SAS: data preparation, linear and
   time-series modeling 
- R: xgboost

Questions for Winners
---------------------

1.	Did you model sales on the log scale?
2.	How did you handle outliers?
3.	What xgboost parms did you use?
4.	What RMSPE scores did you obtain locally?


Thanks for a fun, frustrating (at times), and fulfilling competition!
