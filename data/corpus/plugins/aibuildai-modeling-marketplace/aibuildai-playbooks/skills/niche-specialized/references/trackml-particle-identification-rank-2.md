# 2nd place solution

Competition: trackml-particle-identification
Rank: #2
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63256

In the beginning, I want to build a model which can input all hits and output all tracks. But after simple calculation, it could not be done. So I split it to minimum unit: input two hits. output 1 if two hits are in the same track, 0 otherwise.


The difference with most other DL approaches is that they only do "connect the dots", if some dots lost, the connection break.


**I connect all the dots**. [here is a example of kernel (update 08/16)][1]

In my real case, the difference is: (read the kernel in detail)

 - input size: 27 (x, y, z etc. and use cells to get hit's direction)
 - model size: 5 hidden layers with 4k-2k-2k-2k-1k neurons


The well trained model can get 0.8 by only use the predictions to reconstruct tracks, just like the kernel. Add simple curve fitting (I use scipy.optimize.leastsq to fit circle in xy plane) can get 0.9, and add z-axis constrain (dr/dz) improve 0.003 in the end. I don't spend much time on curve fitting since I think CERN do it better, and someone can get much improvement from better curve fitting.

 - attached is a prediction of event1001 and you may give it a try.
 - fig 01 shows the seed(large circle) and it's corresponding candidates(the same color)
 - fig 02 shows the sum of predict prob. of hits in direct ratio to diameter
![enter image description here][2]

![enter image description here][3]



  [1]: https://www.kaggle.com/outrunner/trackml-2-solution-example
  [2]: https://storage.googleapis.com/kaggle-forum-message-attachments/369970/10058/TrackML_01.png
  [3]: https://storage.googleapis.com/kaggle-forum-message-attachments/369970/10059/TrackML_02.png
