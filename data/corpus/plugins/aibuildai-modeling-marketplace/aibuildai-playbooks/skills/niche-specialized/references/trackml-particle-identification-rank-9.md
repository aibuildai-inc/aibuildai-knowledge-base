# Solution #9

Competition: trackml-particle-identification
Rank: #9
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63250

Did I disclose I have been using DBSCAN?  Well, now you know ;)  

My approach is quite straightforward and has been almost fully disclosed by @yuval.  An helix that starts from z axis can be described by 4 parameters:

 - z at origin `z0`
 - radius `r0`
 - angle at origin in transverse plane (x,y plane) `phi0`
 - slope `zr`

The last 3 can be expressed as functions of the momentum at origin, `px`, `py`, `pz`, and `pt = sqrt(px² + py²)`:

 - `r0 = C * pt`
 - `phi0 = arctan(py/px)`
 - `zr = pz/pt`

The formula to compute `C` is given in the documents from CERN but I estimated it from the train data via a linear regression. That's the only use of supervised learning I made ;)

My code loops over `z0, r0` pairs.  Rather than estimating a distribution I just sample from the tracks in the first 100 train samples.  For each pair I 'unroll the helix', i.e. compute `phi0` and `zr` as:

    phi0 = phi +- theta0
    zr =  (z - z0) / (2 * r0 * theta0)

where `rt` is the distance to origin in transverse plane, `phi` the angle in transverse plane, and `theta0` is the unrolling angle:

    rt = sqrt(x² + y²)
    phi = arctan(y/x)
    theta0 = arcsin(rt / (2 * r0))

In one iteration I add `theta0` to `phi`, and in the next iteration I subtract it.

In order to cope with the discontinuity at `pi` and `-pi` I use `cos(phi0)` and `sin(phi0)`.  

The distribution of `zr` is highly skewed.  Many have use `arctan` to unskew it, but I found that using `arcsinh` was way more effective.  I actually use:

    arcsinh(zr / 0.7) / 3.5

The picture below gives the distribution of arctan(zr), and the one below the distribution of arcsinh. The latter is way more uniform.

![arctan distribution][1]

![arcsinh distribution][2]

The last twist is to model the uneven magnetic field at high values of `z`.  The picture below shows the relative difference between the theoretical angle, and the median of measured angle as a function of `z`, the picture below gives the number of hits in log scale:

![ratio1][3]

The best way I found was to multiply `theta0` with a correction that depends on `z`:

    1.005 - (abs(z + 200) / 6000)**2.4

The picture below shows a smoothed average of the median measured angle deviation (in blue), and my correction function (in red):

![ratio][4]

DBSCAN is run at each iteration.  Its output is merged with existing tracks in a simple way: for each track, or candidate track, I compute the number of volumes with hits from the track.  Hits are then assigned to the candidate track with the most volumes.  Using number of volumes is way more effective than using the number of hits in the track.  

Another criteria is used for deciding which track wins.  I assign a unique `vl_id` to each `volume_id, layer_id` pair.  For each of the first 100 train events I represented each particle track by the sequence of its `vl_id` once data is sorted by `z`.  I then compute the frequency of each sub sequence of 4 `vl_id` .  The quality of each track candidate in test events is computed in a similar way: first create the sequence of its `vl_id` once data is sorted by `z`, then take the average of the log of the frequencies of its sub sequences of length 4, and multiply by the number of volumes of the candidate track.  A hit is assigned to a new candidate track if the new candidate track has both more volumes and  a better quality than the current track of the hit.  This quality is very effective in removing tracks that do not make sense, for instance tracks that skip a layer entirely.   

The above yields a LB score above 0.785 with about 33000 DBSCAN runs.  It takes about 10 hours per event.

The extra mileage I got comes for a simple idea: run another, similar model, on the inner volumes only (7, 8, and 9).  This model can be more conservative (smaller eps for DBSCAN) because tracks are closer to perfect helix.  I ran this model for about the same number of iterations, then merged its output with the previous model: tracks that overlap significantly are merged, and for the rest, the track with most volumes wins.

The very last improvements (about 0.002) come from merging with a third model that is similar to the first one.

That's it, no fancy math, just lots of tuning.  I hope I have not made errors in the equations, I'll check again tomorrow, but appreciate if you find typos.  They must be correct in the code given the results: the code finds about 95% of the centered tracks.

Things I thought about but did not had time to finish implementing:

 - Use direction information from cells data
 - Extend to tracks that do not pass near z axis
 - Fit helix to each track candidate to remove outliers and possibly add missing hits.

I thought my approach would be a nice starting point for second phase, given its simplicity, but I no longer think it is, now that I saw @icecuber solution!


  [1]: https://storage.googleapis.com/kaggle-forum-message-attachments/369949/10056/atan1.png
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/369949/10055/asinh.png
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/369949/10052/ratio1.png
  [4]: https://storage.googleapis.com/kaggle-forum-message-attachments/369949/10053/ratio.png
