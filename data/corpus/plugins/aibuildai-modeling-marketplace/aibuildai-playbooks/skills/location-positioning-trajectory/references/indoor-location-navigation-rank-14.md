# 14th some findings and solution

Competition: indoor-location-navigation
Rank: #14
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/239998

I'd like to thank this great competition's hosts and congrats winners and all medalists!

I think there are so many approach to this problem and I'm busy to read and understand other participants' awesome solusions.

I also write a part of my solution.


1. Findings and preprocessing
There are much more wifi timestamp than waypoints and it is effective to use all wifi data (interpolating waypoint data).
Some BSSIDs show very high-correlated RSSI and getting rid of it is effective.
There some MAC ADRESS for one ibeacon(uuid_major_minor) and we need to use MAC ADRESS instead of uuid.

2. Models
I used LGBM to predict each point indipendently.
In this model, train and predict the waypoints where wifi data exists and predict original WAYPOINT by adding relative posision data.
I used both wifi and ibeacon data.
I used only BSSID and MACADDR which exists both train and test.
These data are transformed to rank.(not rssi value)

3. Postprocessing
Cost minimization with leakage and snap to grid.
In cost minimization, I added leakage term into the original cost function.
In snap to grid, I considered to relative positions (from sensor data) when select the proper grid point.
The notebook is published [here](https://www.kaggle.com/iwatatakuya/snap-to-grid-with-relative-position-data).
In my case, this postprocesing method improved score very much.

4. Ensemble
Ensembles some models and postprocessing.

Thank you for reading this far.
Questions are welcome!
