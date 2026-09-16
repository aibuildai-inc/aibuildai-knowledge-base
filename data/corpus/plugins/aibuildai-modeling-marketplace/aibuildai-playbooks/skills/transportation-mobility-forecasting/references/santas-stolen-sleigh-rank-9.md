# Ninth place solution (12404781212)

Competition: santas-stolen-sleigh
Rank: #9
Source: https://www.kaggle.com/c/santas-stolen-sleigh/discussion/18336

My main optimisation routine was based around a relatively simple algorithm.

**The core components were:**

 * A routine that would place a single gift into the lowest cost position in a group of trips (including if desired an empty trip).
 * A distance metric between gifts that was a weighted mix of haversine distance, closest distance to the same longitude, and distance between longitudes at the equator.
 * A closest_n_trips function that would find a select group of trips in a solution based on the above distance metric (a trip is considered closer if *any* of its contained gifts is closer to the starting gift).
 * A sampling function that would randomly select gifts in one or more trips and remove them to an "unassigned gifts" pool.

**The initial solver was just a random search:**

 * For each gift in turn (from a shuffled list):
   * Find the closest 25 trips (including trip that chosen gift was from). Distance metric weights were 0.025 haversine, 0.85 closest same longitude point, 0.15 equator-equivalent same longitude point.
   * Repeat 500 times for those trips and their contents:
      * Randomly sample from 1 to 3 gifts from each trip
      * Shuffle the array of sampled gifts
      * Re-insert each gift in order, in the optimal position considering the whole group of trips (plus an initially empty trip)
      * Keep the new arrangement if the score is better, otherwise discard.
 * Once every 100 starting gifts:
   * Check all trips in the solution, and split apart into two trips (at a single breakpoint), if that would reduce the overall cost. This seemed to help by preventing trips greedily growing into inefficient combinations.

Running this routine from a starting point of one trip per gift (scoring 29,121,011,015) got a result of 12,422,511,600 in about 4 hours (single thread, and measured on laptop - 2.9 GHz Intel Core i7 - no big compute stack for me!)  I got to this solution by manually tuning the parameters such as distance weightings, number of trips to consider, sample size etc.

**Further improvements were more complex, and progressed more slowly:**

I was aware that the local random searches were likely to miss some important combinations and potentially useful changes. So I constructed a few different searches:

 * A cross-splice search that checked whether two trips could exchange start and end sections to reduce score. I would of liked to generalise this to more splice points, but it was too time consuming to add more degrees of freedom in the search.

 * A "heavy trip breaker" which would take a chosen gift (selected by weight times how much extra "non-necessary" distance led up to it) and construct a new greedy trip from it - stealing gifts from other trips. This was re-optimised using the random search and re-apply algorithm described above; if the ending score reduced across all the selected trips, then the result was kept. Although in practice this only ever found 3 or 4 new trips, I had noticed that the impact of creating those new trips could be very high.

 * A gift search that would take each gift in turn and place it most optimally across the whole solution.

 * A trip destroyer which would remove a trip and try to place all its contents optimally over the rest of the solution. This was kept only if it improved the overall score.

 * A simulated annealing variant of the initial greedy search, typically working on 35 trips with the same distance metric -  I would run 1 million iterations starting from a temperature of 5000 down to 50. Then repeat whole process for another sub-group of 35 trips. Each group of trips would take around 30 minutes to complete.

Of the above, all were useful and contributed to the end score. But by far the most useful in terms of score reduction was the simulated annealing. I suspect it would not have done so well without the wider searches done by the greedy searches though.

Getting the score from 12.422 billion down to 12.412 billion was relatively quick, probably under 2 days' effort. Beyond that point, progress seemed to halve - or worse - each day. I doubt my current code could ever get below 12.404 billion.

**Lessons for next time:**

What I think I should of done next was look into some different moves for continued simulated annealing - either within the local trips, or looking at the whole solution. I think the initial success of the random search and problems with my first attempts at simulated annealing both contributed to me losing time looking at other less productive things first.
