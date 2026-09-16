# 1st place solution

Competition: google-smartphone-decimeter-challenge
Rank: #1
Source: https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262406

First, I would like to thank everyone involved in this competition. It has been a very enjoyable two months. 

Although I am a researcher in the GNSS field, this was the first time I worked with raw GNSS data from smartphones. The raw GNSS data from the smartphone was compared to commercial GNSS receivers used in drones and mobile robots, and I had the following impressions.

- Pseudorange is **very noisy**, and the quality of the GNSS observations is not good
- There are a lot of **missing data**, **duplication**, **time jump**, etc., and it is very complicated

I realized that I was usually exposed to very clean GNSS data...This competition was very very tough because of the huge amount of courses and the variety of smartphones（and the long variable names...）. To be honest, I don't want to look at GNSS data on smartphones anymore!😐

# Key Points for Solution
- Global optimization of position and velocity by **Factor Graph Optimization** Technique
- Velocity constraint by **accumulated delta range（ADR）**
- Absolute position constraint by **differential pseudorange between a base station**
- No machine learning（or rather, it was not possible due to time limitations）

# Input Data
- Phone_Gnsslog.txt
- RINEX files of GNSS[ base station](https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/238579) (I used Verizon base station data from [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=88)）
- Ground Truth（Downtown area only）

In the end, I did not use Phone_derived.csv. There were some missing data in Phone_derived.csv files. I calculated the satellite position and velocity directly from the RINEX navigation file. Baseline position and IMU data are also not used.

# Factor Graph Optimization
> Factor graphs are a class of graphical models in which there are variables and factors. The variables represent unknown quantities in the problem, and the factors represent functions on subsets of the variables. Edges in the factor graph are always between factors and variables, and indicate that a particular factor depends on a particular variable.（from [here](https://gtsam.org/2020/06/01/factor-graphs.html)）

The core of my approach was to use a **global optimization method** based on **Factor Graph**. Several optimization methods have been used [here](https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/261959) and [here](https://www.kaggle.com/c/google-smartphone-decimeter-challenge/discussion/262074), with good results. Factor graph optimization is a method that can apply a variety of complex nonlinear constraints and simultaneously optimize all state variables（the entire driving trajectory）. There are many outliers in the various constraints（edges in the graph）, but a robust optimization technique eliminates the need to manually set the outlier threshold parameters. For more information about factor graphs, see  [this page](https://gtsam.org/tutorials/intro.html).

# Factor Graph Structure
I tried many different graph structures and finally performed Graph optimization using the following Factor graph.
[fgo]

The graph node\\(X_i\\) presents the state variable of different moments, and the edge connecting two nodes presents the error function \\(e(\cdot)\\); each edge corresponds to a single observation \\(Z_i\\). In the graph, error function \\(e(\cdot)\\) represents probabilistic constraints applied to the state at the specified time-step. Optimization factor graph can be written as follows.


Here, \\(\Omega_i\\) is the information matrix（inverse of covariance matrix）which determines the accuracy of the observation \\(Z_i\\). We defined the following as nodes（estimated states）of the graph.



where \\(r\\) and \\(\dot{r}\\) is 3D position/velocity in earth-centered earth-fixed（ECEF）coordinate. \\(t\\) and \\(\dot{t}\\) represent the receive clock bias and drift in each GNSS signals.
 
Here, \\(s\\) in the graph is a [switchable constraint](https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf), which is a state that takes a variable between 0 and 1. The value of switchable constraint is estimated simultaneously by optimization. On the edge of an outlier, the switchable constraint is automatically optimized to 0 and acts like a weight for the observed value. The optimization problem can be described as follows.



where the last term in the above equation is the anchor factor of the switch to prevent the switch state from going to all zeros. 

### Pseudorange Factor
In the Pseudorange Factor, the observation is the difference between the pseudorange and the reference station's pseudorange, which removes the all bias error component in GNSS measurements（excluding multipath errors）. Theoretically, the residuals of the single differenced pseudorange have a Gaussian distribution with zero mean value if receiver clock error is compensated. This single-difference pseudorange improves the absolute position accuracy. Using this differential GNSS technique, I did not need any bias correction for the final position.

### Doppler/ADR Factor
The pseudorange rate（Doppler shift）of the smartphone was noisier than I expected. I think the use of ADR（see P21-P22 of [this document](https://www.euspa.europa.eu/simplecount_pdf/tracker?file=expo/1.1_frank_van_diggelen_-_google.pdf)） led to a higher level of competition. The following figure shows the velocities calculated from Doppler and ADR, respectively, compared to the ground truth. Velocity from ADR is about **5 cm/s**! In terms of accuracy alone, ADR was clearly superior to Doppler shift. Although the ADR is super accurate, its availability is lower than Doppler shift due to cycle slip and half-cycle ambiguity problem, so the Doppler factor is used instead only when the ADR factor is not available.

[adr]

### Motion Factor
The motion factor uses the estimated velocity and clock drift to add constraints between neighboring nodes. It simply adds a constraint so that the integral of the velocity and clock drift is equal to the difference between the neighboring states.

### Pseudo-Position Factor
The pseudo-position factor was used only for the downtown area, and since the ground truth was traveling along the same path as the test data, the points on the ground truth path closest to the estimated trajectory were extracted and added to the graph as pseudo position constraints with appropriate covariance.

# A Few Concerns
- As you can see in this discussion, the CV and LB scores are not matched, as is the big difference between the public and private LB scores. Without machine learning, I don't think there would be much overfitting to public scores. I can't explain the difference in these scores.

- I have some doubts about the accuracy of Ground Truth. The ground truth is based on Novatel SPAN（I also have this sensor）, but the results are quite different depending on multipath situations and analysis parameters. I would like the organizer to release the ground truth of the test data for future evaluation. I would also like to know the variance of the estimated ground truth.

# My Impressions
- Too bad I failed the decimeter challenge. The best public score in all my submissions was 1.4 m. I failed to choose the final post... I am convinced that with a combination of different techniques, I can eventually break 1 m!😃

- Graph optimization was very powerful. It optimizes all variables at the same time under nonlinear constraints, so it performs much better than filtering methods such as KF. For the implementation, I used the [GTSAM](https://github.com/borglab/gtsam), which is C++ library, and has Matlab and Python wrapper.

- It's a shame that I couldn't incorporate machine learning, which I wanted to do at first. I'm enjoying reading the other team's solutions. Let's write a paper together.

- There was a considerable difference in the GNSS observations depending on the phone. In the end, I didn't use phone marge and estimated the trajectory of only the best phone for each run. My best smartphone is Samsung Galaxy S20 Ultra, which is very good at tracking the GNSS carrier phase. I love it😊

- The only thing I could trust was ADR. Thank you so much, ADR.
