# 14th place solution

Competition: m5-forecasting-accuracy
Rank: #14
Source: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163211

Thanks to everybody for a great competition. 
Congratulations to the prize winners. 
I got my first solo gold as my first medal in kaggle.

The outline of my solution is as follows:

1.Model:  LGBM for each [store__id] x [dept__id] ( totally, 70 models )
- Objective: tweedie (custom object do not improve CV)
- Weight: cumulative actual dollar sales (last 28 days)
   ( I do not use scale for RMSSE. )

2.Data period: 2014/1/1~ (before 2014, the trends are different)

3.Features
- Lag demand
   28 days shift + mean and std for 7, 14, 28 days, etc..
   (day by day for 1~7, 14, 21 days, and recursive : I use one with lower CV for each [store__id] x [dept__id])
- Sells prices
   some statistics of sells prices and, sell prices for same item__id in CA_3
   (Sell prices for same item_id in CA_3 explain zero sales period for some items.)
- Calendar
    Target encoding for ids (store, item, ...) x [weekday, events, or month] or ids with last 3M, 1Y, 2Y (rolling, no leak).
     This variables works well. 
      Additionally ,I remove some variables following permutation importance.
    
4.CV: last 3 x 28 days, and (sub : last 2 x 28 days + same period at 1 year before)
