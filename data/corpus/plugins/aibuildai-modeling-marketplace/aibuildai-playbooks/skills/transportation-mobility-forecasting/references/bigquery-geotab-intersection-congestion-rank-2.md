# 2nd place solution

Competition: bigquery-geotab-intersection-congestion
Rank: #2
Source: https://www.kaggle.com/c/bigquery-geotab-intersection-congestion/discussion/122314

First of all, many thanks to the organizers for putting together this very interesting dataset and for setting up this competition! It gave us the opportunity to explore several approaches and find some ideas for further work in the future.

**tldr:** We used a combination of three tree-based algorithms, extensive feature engineering, limited external data and almost no transport modelling. We developed a different set of models for each of the four cities.

**The problem statement**: We had to predict the variation in congestion at different points in space and time, using information from a number of examples covering a subset of those points. So, in fact there were two underlying problems: 
a. To interpolate/extrapolate congestion predictions for specific intersections using data from the same intersection at other points in time  
b. To predict congestion in intersections we don't have any information about

**Our solution**:
The models should obviously avoid using IntersectionId or any other information related to specific intersections that would not allow the model to generalize. We built several features that explored the setup of each intersection and the relevant role of the path concerned (e.g. number of possible paths &amp; turns, number of different possible entries &amp; exits, type of turn (especially left turns / same entry-same exit directions + same name) and tried to "solve" the network by identifying the previous/ next intersection each specific path is connected to. This in turn allowed us to calculate the length of the link between intersections which proved very important especially for the trickiest of the six congestion variables to model, the p80 of Total Distance.

We also tried to "solve" the network using graph theory indicators. We transformed the IntersectionId set into a graph and calculated the betweenness indicator for each node. In theory this corresponds to the number of shortest paths in the network that pass through the node. But since we don't have information about the overall distribution of trips, in practice this was very close to a proxy of the geographic centre of the network. It ended up having a low impact in our models.

We avoided using coordinates and direction as stand-alone features to avoid over-fitting. Instead we used k-means to identify a number of clusters and calculated the distances of each intersection to the cluster centres. For example, if k=1 we have the distance to the city centre. The best models used zip code population as the clustering and a k value of between 6 and 10. Clusters based on either p50 variables also seemed to work.

We were positively surprised by the quality and precision of the data included in the TIGER data. With only a little processing, one can easily match the competition data to the real networks. However, on their own, the TIGER categorizations (type of road mainly) do not really help in modelling congestion. We also tried using actual traffic data which is publicly available, but it didn't help. In one city the latest data was from 2006 and in another we didn't find a way to extract a csv or shapefile from the API. Anyway, we managed to link average traffic counts with points on the network and from there tried to estimate traffic counts for the paths of each IntersectionId, with very limited success. But we think that it can be eventually done, with enough time and dedication... 

We feel that we could have gone even further, but at some point it seemed we had done enough, our solution was far ahead in the public LB. Apparently it was not ;), we were surpassed by Peter in the private PB! We have another solution (that was not selected for the final evaluation) scoring 59.97 and several half-baked models that can probably improve the score even more.

**Additional value from data**: We really appreciated the data on the distribution of time and distance (the various P.x0). While they were of no use for the competition itself, they can provide useful information for practitioners. We didn't try going the classification way and try to predict which congestion profile every path could have (it would take too long for this competition and theory says that it is difficult to model the distribution itself) but we may try it later on ion our own.

**External data**: We only used data available in the competition and the external data declared. We know that there is private data on traffic counts, speeds and congestion data that could have been very useful for this problem, but we didn't have or use any. (We have good data and models for Europe, but not for the US) 

**BigQuery**: We made a few attempts to use BigQuery but the setup to access BQ was too complicated and we resigned quickly. In principle, SQL approaches should be useful for this type of problem and in fact we know of several application in the domain that use them for data handling and processing. From what we understand from the discussions however, the weakness of BQ for this competition is the lack of variety in models it can accommodate.
