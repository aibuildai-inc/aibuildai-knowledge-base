# Window level regression

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #67
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/35397

Hi all,

For anyone interested here's an approach that extends mrgloom's image-level regressions script to work at the window level.

It extracts 512x512 windows from the images, counts the lions using the dots and blob detection, and compiles a significantly enlarged training set. 

Prediction is done using a sliding window. Any incomplete edges are dropped, which isn't ideal

It's not totally finished, never really scored that well, and won't be as interesting as I'm sure the winning solutions will be, but I think it's different to the other approaches posted so far (?). 

Enjoy!

https://github.com/garethjns/Kaggle-Sea-Lions-Solution/tree/master
