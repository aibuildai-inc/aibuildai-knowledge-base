# 3rd place solution

Competition: trackml-particle-identification
Rank: #3
Source: https://www.kaggle.com/c/trackml-particle-identification/discussion/63330

Hello everyone and thank you for the nice competition!


Unfortunately I joined it late and didn't follow discussions on the forum, I had to concentrate on solving the task.

I developed a combinatorial algorithm, which is very similar to the 1-st place algorithm from icecuber. 

In the amount of time I had I only managed to finish a combinatorial "engine" of the algorithm a day before the deadline. During the last day I just tried to run this engine many times more or less chaotically  everywhere in the detector. The resulting code looks therefore horrible and is extremely slow. But the engine itself seems to work, the rest still needs to be developed.


Ok. The engine consist of the two parts: 

-the first part converts hits to short tracklets in some local part of the detector. It's goal is to kill the hit-to-hit combinatorics in order to work later on on some structured data.

-the second part prolongates the tracklets through the detector and collects their hits. 


I run this engine several times in different parts of the detector with different constraints.
Then I sort all the found track candidates in some simple way and choose the best tracks. Hits which belong to the found tracks I remove from the plane. Then the next round of the tracklet search starts in other detector parts or with other constraints.
This way I clean up the data slice by slice, so to say, until nothing is left on the plane. 


The engine. 


1. The tracklet constructor.

There are 2 options here. 

option a) 
It creates two-hit tracklets which are constrained to the event vertex (which is pretty much (0,0) point in XY and +-2.5 cm in Z )
Using the vertex constraint is the trick here. It significantly reduces amount of possible tracklet candidates. And, as 75% of the tracks are coming from the vertex, one can really clean up the data by removing the vertex tracks before doing anything else.

option b) 
It creates 3-hit tracklets without vertex constraint.  

The layers where the tracklets are created are predefined by the main program. Unlike icecuber I decided to not create all possible tracklets everywhere but rather save the compute time by developing some smart seeding sequence strategy, which is still need to be developed:) 


2. The tracklet prolongation. Similar to the winner algorithm, I don't have a global trajectory. 

To prolongate a track to the next layer, I use a local helix created by last 3 hits of the track. 

Amazingly, it works pretty well. I think this is due to very precise measurements in silicon. One can follow all the local features of magnetic field and trajectory scattering in the material, and complicated energy losses, and god knows what else without even knowing the value of the magnetic field!  And it doesn't cost any cpu time.

But what I found, the magnetic field is varying  dramatically in the detector, from 20 kGaus to -10!!. Especially between the detector volumes. I realised it when I was checking - how much the local curvature changes along a track. Oh yes, it changes. 

To investigate this, I have fitted the magnetic field value using neighbouring truth points and truth momentum vectors. 

Once I realised that the field is non-constant in many regions, I decided to modify my track model. 

It is still a helix, but it is parameterized not with its geometrical radius, but with a physical parameter Pt (transverse momentum). 
These are just proportional: Pt = B*r.   Now, having 1) the physical parameterisation and 2) the magnetic field values, I can fit the trajectory with one field value (i.e. inside a radial volume), but then prolongate it using another field value (i.e the value between radial and forward volumes). 

Or, in the other words, I scale the helix radius  according to the field change. (New Radius=OldRadius*NewField/OldField)

The magnetic field I parametrised for each detector layer individually using some polynoms. 
Here is my formula for the field: 
B(z,phi) = (c0+c1*z) + (c2+c3*z)*sin(phi) + (c4+c5*z)*cos(phi).
Here z, phi are  hit angular and z coordinates on a layer, coefficients I have fitted with old good LSM method. The approximation is  not very accurate, maybe one can replace it with just an average field value on a layer.  To save the time during track search, I calculate the filed value for every hit and store it in the hit structure before the search starts. 

As I remember, use of the physical model improved my accuracy of prolongation, but I can't tell now how big was the improvement. May be at the end it is not needed at all.

One big problem I have here - I have to manually set cuts for picking up hits. For the proof-of-concept it was fine, but then I wind up with a huge file with copy-pasted and slightly modified hardcoded numbers.  My plan was to have this hits picking-up cuts to be set automatically from the test data. But due to the naive trajectory model, hit deviations from  trajectories are not nicely distributed. At the end I had to look to every distribution and decide where to cut it. One should do something about that.


The search strategy. 

First, I search tracks which are coming from the vertex, then the other tracks. First I find high-momentum tracks (applying angular and momentum cuts on the tracklets), then the low-momentum tracks. 


Well, that is pretty much the algorithm. 

Thank again for the nice competition!

Here is a link to the code: https://github.com/sgorbuno/TrackML_CombinatorialTracker
Here is a description: https://github.com/sgorbuno/TrackML_CombinatorialTracker/blob/master/doc/TrackML_AlgorithmDescription.pdf
