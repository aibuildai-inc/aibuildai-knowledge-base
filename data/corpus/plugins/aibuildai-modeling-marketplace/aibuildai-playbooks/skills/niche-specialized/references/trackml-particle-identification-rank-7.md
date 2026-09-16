# 7# solution

Competition: trackml-particle-identification
Rank: #7
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63313

Although I shared many of  our algorithm's basic ideas, now is the time to share the full solution
I have posted a kernel with the major part of this solution. [#7 place solution][1]
Running this kernel on training event 1000 will score ~0.635 after the clustering stage, and 0.735 after expending stage.
Every stage takes about 8-10 min on Kaggle, and about half the time on my laptop.
In the clustering part of the kernel the algorithm 5500 pairs of z0, 1/2R (More of it below) by increasing the number to about 100,000 the score will plateau at about 0.765 (after expending).
How does it work:
In each clustering loop the algorithm try to find all tracks originating from `(0,0,z0)` and with a radius of `1/(2*kt).`
If a hit (x,y,z) is on a track the helix can be fully defined by the following features (1), (2)

    rr=(x**2+y**2)**0.5
    theta_=arctan(y/x)
    dtheta = arcsin(kt*rr)
    (1)	Theta=theta_+dtheta
    (2)	(z-z0)*kt/dtheta

To solve the +pi,-pi problem we use sin, cos for theta.
 To make (2) more uniform, we use `arctan((z-z0)/(3.3*dtheta/kt))`
After calculating the features, the algorithm tries to cluster all the hits with the same features. This is done by sparse binning – using np.unique.
The disadvantage of sparse binning over dbscan is it’s sensitivity, the advantages are its speed and its sensitivity (almost no outliners).
After clustering every hit choose if his cluster is good according to the clusters length.
Every 500 loops all hits belonging to tracks which are long enough are removed from the dataset
If two hits from the same detector are on the same track, the one which is closest to the track’s center of mass is chosen.
The z0, kt pairs a chosen randomly
While running, the algorithm changes the bin width and the length of the minimum track to be extracted from the dataset.

Expending is done by selecting the un-clustered hits which are close to the center of mass of the track.

To get better then 0.765, we merged a few long runs together, this was done by scoring the tracks with a ML algorithm Trian wrote (please share below).
We also gained sum points by clustering from outside of the origin, starting the track from a hit (it was very efficient and slow)


  [1]: https://www.kaggle.com/yuval6967/7th-place-clustering-and-extending
