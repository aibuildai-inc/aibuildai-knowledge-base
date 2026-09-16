# 16th place Solution

Competition: playground-series-s3e5
Rank: #16
Source: https://www.kaggle.com/c/playground-series-s3e5/discussion/386745

Thanks to the organisers for another challenging and interesting competition.

I tried a  number of different things, of which probably the most interesting was to rank the test instances from highest to lowest predicted class, using real numbers rather than simply integers, and then to fit this to the expected distribution based on the {training plus original} sets.

In the {train plus original} data, once duplicates are removed, we have quality values as follows:

8: 1.8% 7: 14.8% 6: 38.6% 5: 41.4% 4: 2.9% 3: 0.6%

We import a file containing integers in these proportions, the same number of integers as there are in the test set. Then we match the top 1.8% of our wines to category 8, the next 14.8% to category 7, and so forth.

I implemented this [here](https://www.kaggle.com/code/jbomitchell/boltzmann-ensemble-match-to-ideal-distribution).

Based on the success other people were having with simple ensembles, I also tried using a simple median of top public solutions [here](https://www.kaggle.com/code/jbomitchell/median-ensembler).

Ultimately, the generally poorer public LB scores from the idealised matching persuaded me to select two of the median-based submissions, which turned out not to be optimal. In fact, I had even a [simple ensemble of two models](https://www.kaggle.com/code/jbomitchell/simple-blend) that would have placed sixth had I known to select it.
