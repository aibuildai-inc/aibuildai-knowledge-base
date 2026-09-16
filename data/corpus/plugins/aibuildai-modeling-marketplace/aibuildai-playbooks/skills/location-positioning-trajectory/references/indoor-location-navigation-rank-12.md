# 12th place solution

Competition: indoor-location-navigation
Rank: #12
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240196

Congratulations to all the winners. Even if I had no experience in the domain, I found this competition very interesting and decided to enter it very early. This allowed me to try different ideas. Let me present the one that I used in my final submission.

**1 - Pre-processing**
I used only wifi signals for positions prediction. Beacon or magnetic data didn't help.
I grouped all wifi rows by block, but I reallocated wifi rows based on "last seen timestamp" to the wifi block that was the closest with respect to timestamp.

**2 - Floor prediction and Wifi-based position prediction**
In a first step, I used LGBM models for floor prediction (1 model by building) and simple 2 layers feed forward NNs for (x, y) positions (1 model for each floor). The performance was OK.
Then, I switch to a rather radical approach that was providing better results:
I computed for each test point the cosine of the angle of wifi fingerprints with respect to each training point. That is: the scal prod divided by the norm of the 2 vectors.
Using the cosine instead of the actual scal prod is important because the intensity of the wifi signals varies a lot for different positions.
As a result, a perfect match would return a cosine of 1, and the value decreases as the closest training point "match" is farther away (actually, there is also a bit of post-processing to discard outlayers).
For each test point, the best training point "match" is computed FOR EACH FLOOR.
The floor prediction is then performed by checking the evolution of the cosine along the path: the floor for which the cosine is the highest more frequently is selected as the predicted floor.

Using this approach (which is a kind of k-NN), there is no model (no parameter to train), and no CV...
Actually, following the simple idea presented in the thread: [https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization](url), I submitted an edited version of one of my final submissions and was happy to see that my floor prediction was 100% accurate both for public and private data:

| Submission and Description | Private Score | Public Score |
| --- | --- | --- |
| submission_df_leak_start_end - REFINED_PATHS-FullPath-SE-NoENF-LS_submission-2021-05-17_07-58-29 - score2.692.csv | 3.25303 | 2.69255 |
| submission_INCREMENT_2021-05-18_17-42-06.csv | 18.25303 | 17.69255 |

**3 - Introducing accelerometer data**
With the approch discussed in section 2, the position prediction is still very noisy. It is necessary to exploit the accelerometer data.
At this stage, I didn't start from the raw data but I used instead the library provided by the organisers ("compute_step_positions").
First, I used my own post-processing (using some local averaging along paths, combining initial (x, y) predictions and step positions using a moving window). However, Saito's notebook was much more efficient. Also, the weights in the cost minimization expression could use the cosine values to reflect points for which one is more confident.

**4 - Improving paths by local search**
The output of section 3 is a list of (x, y) positions that define paths.
I defined another cost function that combined 3 values for each path:
   - error for (x, y) positions (current positions vs "starting" positions, for each point on the path)
   - error for length of each segment in the path (ratio between length based on start/end positions, and associated step_positions length)
   - error for heading between consecutive segments (again: variation in heading based on (x, y) positions vs variations in heading based on accelerometer data)

Then, a population-based search was performed using some "mutation" operators to update the path and capture constraints to place points in corridors.
Again, no CV framework was used here, but improvements in the local search based on this cost function appeared highly correlated with LB scores.

**5 - Post-processing**
I didn't explore that part that much. I just lazily invoked the snap-to-grid ([https://www.kaggle.com/dragonzhang/3-3-g6-indoor-navigation-snap-to-grid](url)) and leakage ([https://www.kaggle.com/tomooinubushi/postprocessing-based-on-leakage](url)) notebooks like many of us :-).
These notebooks were regularly improving my solutions by a value of around 0.3.

**Final comments**
One direction I wanted to investigate was about correcting step_positions returned by the python library provided by the competition host. Indeed, doing some statistics with training data, one could notice significant discrepancies between segments based on actual training positions and corresponding step_positions (roughly, with a factor in the range: 0.5 to 1.5, depending on paths).

I think the main benefit of the approach presented above is that it does not rely too much on training points (except for final snap to grid). I believe this makes it more applicable to real life applications for which positions to predict don't match known positions... However, it does not take advantage enough of the setup of this competition for which a large fraction of test positions matches training positions!

**References:**
I want to thank the authors of the following 4 notebooks that helped me a lot:
1) [https://www.kaggle.com/saitodevel01/indoor-post-processing-by-cost-minimization](url)
Like many of us, I also used Saito's code in my approach. At the beginning, this notebook was not improving my solution that much, but runtime was much better!
2) [https://www.kaggle.com/rafaelcartenet/scaled-floors-geojsons-new-dataset](url)
I wasn't familiar with the shapely library. This notebook provided me with all the code for supporting the local search approach that I implemented. Many thanks to Rafael Cartenet!
3) [https://www.kaggle.com/dragonzhang/3-3-g6-indoor-navigation-snap-to-grid](url)
4) [https://www.kaggle.com/tomooinubushi/postprocessing-based-on-leakage](url)
