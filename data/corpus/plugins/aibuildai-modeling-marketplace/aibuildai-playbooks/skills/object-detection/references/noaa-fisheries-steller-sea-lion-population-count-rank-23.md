# 23 th place solution

Competition: noaa-fisheries-steller-sea-lion-population-count
Rank: #23
Source: https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/36066

Hi Everyone,

Ending final validation, I share my solution.<br> 
My approach is based on semantic segmentation following by simple regression.<br>
I remove full-connect layer from resnet 34, and use 7x7 buttom surface for segmentation.

Code: https://github.com/toshi-k/kaggle-steller-sea-lion-population-count

![enter image description here][1]


  [1]: https://raw.githubusercontent.com/toshi-k/kaggle-steller-sea-lion-population-count/master/img/solution.png
