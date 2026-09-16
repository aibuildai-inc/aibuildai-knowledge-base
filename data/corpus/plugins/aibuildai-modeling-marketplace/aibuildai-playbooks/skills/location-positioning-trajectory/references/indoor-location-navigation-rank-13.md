# Super simple and super effective trick - 13th place

Competition: indoor-location-navigation
Rank: #13
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/239884

Thank you all for the competition!

Organizers provided us with functions which enabled computing vectors of movement (x and y coordinates). Average error ((x_hat - x) ^ 2 + (y_hat - y) ^ 2) ^ 0.5 was **2.67m**. I have build a MLP which corrected this calculation and brought the error down to **1.86m**. The only input was: **building** one hot vector, **floor** one hot vector and **raw calculations**! Even without building and floor (with just row calculations!) the error could be brought down by around 25%.

It boosted the cost minimization. Later I have used tetris-like fitting paths into hallways. I am going to share more in the following days.
