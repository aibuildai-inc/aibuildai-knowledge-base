# My approach - public 0.507, private 0.517

Competition: favorita-grocery-sales-forecasting
Rank: #15
Source: https://www.kaggle.com/c/favorita-grocery-sales-forecasting/discussion/47534

![dd][1]

# Models
I use two models to solve this problem

1. LightGBM, it's based on Ceshine lee's script, but I add some other new features.
a) zero crossing rate (https://en.wikipedia.org/wiki/Zero-crossing_rate), I create this feature for different windows(1, 3, 7, 14, 30, 60) and levels(item level, store*family level)
below is my implementation  

    def corss_zero(row):
          return (((row[:-1] * row[1:]) == 0) * ~(row[:-1] == row[1:])).sum()

b) zero-sales counts in different windows  

    "zeros_28_2017": (get_timespan(df_2017, t2017, 28, 28) == 0).sum(axis=1).values  
        "zeros_56_2017": (get_timespan(df_2017, t2017, 56, 56) == 0).sum(axis=1).values  
        "zeros_84_2017": (get_timespan(df_2017, t2017, 84, 84) == 0).sum(axis=1).values 

b) lastest promo day in last 30 days  

    def fo_in_row(row):  
    j = 0  
    for i in row:  
        j += 1  
        if i == 1:  
            return j  
    return -1  
"lastest_promo_30": get_timespan(promo_2017, t2017, 30, 30).apply(lambda row: fo_in_row(row[::-1]), axis=1).values.ravel()    
2. NN
I used the simplest feed-forward neural network.
same features as LightGBM, for pre-processing, I used ”RankGauss“ recommended by @Michael Jahrer(https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629), for training I used BatchNorman, big dropout and nothing else.

small tricks.
I separate Perishable and Non-Perishable variable to train different models. and this give me a little improvement.

# Validation Framework
I used the same framework recommended by @Ceshine lee, but I paid careful attention to first 5 days and last 11 days, I only add variables that both have improvement to first 5 days, last 11 days and Public Board, this give me a rather steady improvement on both public and private. 


  [1]: http://imglf5.nosdn.127.net/img/OEUzbU5jenB4YlRsQWNNcnk3Um5DVnFkc0Y0eEQyMFQzRXRlRXRBWW5NakMyU28yQnc4MS9nPT0.jpeg?imageView&amp;thumbnail=2086y226&amp;type=jpg&amp;quality=96&amp;stripmeta=0&amp;type=jpg
