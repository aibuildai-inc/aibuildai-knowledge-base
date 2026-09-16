# [11th] LSTM + Dynamic Programming Post-processing

Competition: indoor-location-navigation
Rank: #11
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/239981

Congratulations to all the winners! And specially to my teammate @Ouranos, that becomes also Grand Master after this very interesting competition. We team up in the last week and that give us an amazing boost. At the end something went wrong, but we get our gold medal 😊

My work has been mainly in the solution postprocessing.  The question was how to take into account simultaneously at path level, a **regression model** for the positions, the relative information from the **sensors**, the sequence of paths obtained from identifying unique **devices**, and most importantly, the fact that the positions are discretized over a **grid**.

The funny thing is that is possible to express that as a **combinatorial optimization problem** over the grid, with a cost function that express that we want a solution over the grid, as near as possible to the regression model, that agree as much as possible with the deltas obtained with the sensors, and use the information of the sequence of paths all simultaneously
For a finite grid of points Xg and a given path p with nt steps, with model X(t), sensor relative position Delta(t), information of previous and next path Xpr,Deltat_p, Xn,DeltaT_n,  we can write a cost function:

$$
\sum_{i=1}^{nt}  \epsilon (Xg(t)-X(t))^2 + \alpha   (Xg(t-1)-Xg(t) + Delta(t))^2 + \gamma  (Xg(1)-Xp)^2/Deltat_p  + \gamma (Xg(nt)-Xn)^2)/Deltat_n
$$

Given suitable alpha, gamma and epsilon this expression can be optimized **exactly** using **dynamic programming** over t.

In our case, the model was developed by @Ouranos. It is an average of 3 LSTM models, one of them specific for every building,  trained with pseudolabeling 

Train -> post-process-> Pseudolabel train-> post-process

 The deltas are calculated using the code provided by the organizers in their github, and the sequence is obtained in a different way that has been described in the discussions.  We use a hash of the device information provided in the .txt files 

```
#	Brand:OPPO	Model:PBCM10	AndroidName:8.1.0	APILevel:27	
#	type:1	name:BMI160 Accelerometer	version:2062600	vendor:BOSCH	resolution:0.0023956299	power:0.18	maximumRange:39.22661
#	type:4	name:BMI160 Gyroscope	version:2062600	vendor:BOSCH	resolution:0.0010681152	power:0.9	maximumRange:34.906586
#	type:2	name:AK09911 Magnetometer	version:1	vendor:AKM	resolution:0.5996704	power:2.4	maximumRange:4911.9995
#	type:35	name:BMI160 Accelerometer Uncalibrated	version:2062600	vendor:BOSCH	resolution:0.0023956299	power:0.18	maximumRange:39.22661
#	type:16	name:BMI160 Gyroscope Uncalibrated	version:2062600	vendor:BOSCH	resolution:0.0010681152	power:0.9	maximumRange:34.906586
#	type:14	name:AK09911 Magnetometer Uncalibrated	version:1	vendor:AKM	resolution:0.5996704	power:2.4	maximumRange:4911.9995
#	VersionName:v20191120-nightly-9-gde3748b	VersionCode:424	
```

When no information is provided, the 5 first characters of the path is a good proxy.

Before starting the optimization, we extend the original grid dynamically with 2 types of points:
1.- We add points from the X solution that are far from any original grid point.
2.-When the distance between points in the X solution is bigger than a certain threshold, we introduce intermediate points.

We are sharing the post-processing code at [11th-dynamic-programming-post-procesing] (https://www.kaggle.com/vicensgaitan/11th-dynamic-programming-post-procesing)
