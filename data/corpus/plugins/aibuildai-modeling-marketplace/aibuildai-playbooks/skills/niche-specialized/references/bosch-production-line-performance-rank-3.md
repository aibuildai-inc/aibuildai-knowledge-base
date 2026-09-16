# #3 place solution

Competition: bosch-production-line-performance
Rank: #3
Source: https://www.kaggle.com/c/bosch-production-line-performance/discussion/25359

Before starting... a big thanks to Bosch team and all of the folks who contributed a lot to the community's learning in this competition –  among others, JohnM's (superb!) and gingermans' flowpath visualisations; CPMP and Daniel FG's super fast MCC measurement functions for R and python; Mathias (Faron) and Marios for releasing the leak early and making it a competition; all the wrapper's for lightgbm (I used Raddar's R functions, not to take away from great ones by Laurae(great top10 finish!), Aradalan and others – and Microsoft team for releasing it).   
Congrats to Gabor, Ash, Daniel FG, scnndl (looking forward to all your solutions!) and all other players.... and of course my team mates Marios, Mathias and Stas! I told Marios at the start I wanted to get GM status from this competition, and to the end he was determined to keep to that :)

For feature engineering, here is what worked roughly in order of effectiveness,

- Mathias's leak shown in the [public script][1] 
- Second leak, used per station, which worked especially well on L3_S29 and L3_S30. If you check JohnM's [viz][2], you can see these lines have very few NA's so the order by date can be captured very well. On other lines, with more NA's, the date order is more broken. We basically sorted on the dates in each station, and put an indicator for sequential duplicate numeric rows within the station. Created a leak feature like this for each of the stations, but L3_S29 and L3_S30 worked particularly well.  
- Related to second leak above, within each station, column bind the numeric values to a string, and get the count of the unique strings.     
- Row-wise summed the count and the leak (found in the last two bullet points), over all the stations.    
- Out of fold rolling mean of target (using 2 fold to replicate train/test split); sorted based on start and end dates of all stations. Got different rolling mean windows – 5, 10, 20, 100, 1000, 5000 – interestingly it was the really big windows which helped most; probably because it captured periods with low failure rate. 
- Lag and lead of target out of fold.   
- Encoded a few categorical columns with out of fold Bayesian Mean -there was one cat var with a lot of levels which captured a lot of 1's particularly well. This encoding was limited to the non duplicated categoricals which had over circa 7 unique values.  
- Counts of non-duplicated categoricals columns, counts of non-duplicated date and numeric columns.   
- Encode the paths of whether each row passed through stations.  
- Row wise NA counts for numeric, as well as max/min per stations.   

We had about 50 models in a stack, made up at second level of xgb, neural net.. and I think linear models. First level models were xgb, lightgbm, extratrees, rf, glmnet, neural net and others. Best single model was lightgbm with LB 0.502. As I guess everyone experienced, the stack was very instable, which lead to a lot of frustration and long periods of stagnation. 

We had an unplanned boost in the last hour of the competition :) With the instability, as BreakfastPriate pointed out, there seemed to be a few 100 rows being the main drivers for the last bit of score. We downloaded around 25 subs (containing the 1's and 0's) from LB which scored over 0.51, then column binded these subs and took the median value, row wise, of the 25 Response values. This gave a nice boost - around 0.003 on public and private – moved us up to fourth place on public LB .... with 20 mins to go. Had 2 more subs, and tried different thresholds, as opposed to median, but this didn't improve the straight median of all the 25 best subs.   
A histogram (cut off on the y-axis) of the row wise sum of the 25 subs can be seen below...   

![enter image description here][3]


  [1]: https://www.kaggle.com/mmueller/bosch-production-line-performance/road-2-0-4
  [2]: https://www.kaggle.com/jpmiller/bosch-production-line-performance/flowpath-viz/code
  [3]: https://www.dropbox.com/s/z9b91qrh2cg0mhy/25subs.png?dl=1
