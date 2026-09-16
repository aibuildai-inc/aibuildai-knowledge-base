# 3rd place solution

Competition: google-smartphone-decimeter-challenge
Rank: #3
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262355

Looks like I should somehow present my solution

First, I want to beg your pardon for my terrible English, but hopefully there will not be a much text.

I did not use any preprocessing or post-processing for the data, with one exception - I had to filter DeltaRange because on some tracks they are completely false. Also, I have not used any complex algorithms.

All I used from the data is pseudo-ranges and deltas.

In order to filter out the deltas, I calculated the position offset to the next epoch, and left only those epochs that gave enough consistent satellites

The main idea of my algorithm was that the positions that I get on the track must be in good agreement with the psevdoranges and deltas, in addition, they must satisfy certain physical laws.

To do this, I built a tensorflow model that contained positions and time offsets as weights. Then I minimized the loss, which consisted of the sum satellite distance errors and the delta changes errors. Plus, I add penalty for unnecessary acceleration - to avoid trajectory wobbling.

This allowed me to get the first place in the public rating and the third in the private one.
As you can see, I did not use the data from the ground true positions at all.

After the end of the competition, I spent a little bit more time and added psevdoranges correction from the base station and the calculated offset relative to the gt positions, that greatly improved the result and now it looks like this
Private  Public 
1.39533 2.59287

On training data, the error decreased to 1.1

I'm going to add some more data from the IMU - perhaps this will allow us to overcome the error of one meter
