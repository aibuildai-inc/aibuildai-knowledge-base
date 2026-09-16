# 1st Place Solution Summary

Competition: expedia-hotel-recommendations
Rank: #1
Source: https://www.kaggle.com/c/expedia-hotel-recommendations/discussion/21607

I'd like to note that I'll be donating 10% of the prize to the American Cancer Society in honor of [Lucas][1].  He was always an inspiration to me here on Kaggle.

In terms of the solution, let me describe a couple of the components before getting into the model.

 - Distance Matrix Completion

----
The idea was to map users and hotels to locations on a sphere.  If we can do this successfully, then we can 'widen the leak' not just to previously occurring distances but to potentially any example where the distance between user and hotel is known.

One immediate issue with this approach is the question of what combinations of columns to use in uniquely identifying user and hotel locations.  For users, the tuple U=(user_location_country, user_location_region, user_location_city) was a natural choice.  For hotels things are less clear.   Two parallel notions of hotel location were developed H1=(hotel_country, hotel_market, hotel_cluster) and H2=(srch_destination_id, hotel_cluster).  Both the H1 and H2 versions were used as features and the final model was relied upon to sort out which was more useful.

For both H1 and H2, user and hotel locations were randomly initialized on a sphere and gradient descent was applied to the [spherical law of cosines formula][2] on the distinct combinations of (U, H, orig_destination_distance).  

Convergence for the gradient descent was not quick at all.  Using nesterov momentum and a gradual transition from squared error to absolute error, the process took about 10^11 iterations and 36 hours.

In the end, the average errors for H1 and H2 were around 1.8 and 3.7 miles respectively and I'm looking forward to finding out what tricks the other teams had here.

 - Factorization Machines

---
These tend to pick up on interactions between categorical variables with a large number of distinct values.  The hope was that they would find some user-hotel preferences in-line with Expedia's description of the problem.

The implementation used was [LIBFFM][3].  For each of the 100 hotel clusters, a seperate factorization machine model was built using the categorical features in the training set.  The attributes for each model were the same but the target for k-th  model was an indicator of whether the hotel cluster was equal to cluster k.

These FM submodels added about 0.002 to the validation map@5.  Aside from leak related features, these provided the most lift over the base click and book rates.

 - Training/Validation Setup

---

The strategy here was to build a model on the last 6 months of 2014, from '2014-07-01' onwards.  In order to mirror the situation in the test set, the features used to train the model were generated using only hotel cluster and click data prior  to '2014-07-01'.  

When scoring the model for the test set, all the features were refreshed using all available training data.

A portion of the user_ids from '2014-07-01' onward was set aside as validation. This did not line up precisely with the leaderboard, but it probably agrees directionally. 

 - Learn-to-Rank Model

---
In order to turn this into an xgboost "rank:pairwise" problem, each booking in the training sample was "burst" into 100 rows in the training set.  The features for each row were generated relative to the corresponding cluster.

The features input into the model were essentially the historical book and click rates, distances derived from the matrix completion, and the factorization machine scores.  So, the ***i***-th row corresponding to the ***j***-th booking would be something like:  

 - the historical click and book rates of cluster ***i*** for someone having the attributes of the ***j***-th booking
 - the difference between the matrix completion predicted distance and the distance provided for the ***j***-th booking
 - the factorization machine score for the ***i***-th cluster based on the attributes of the ***j***-th booking.

---

For the final scoring, the test set is also "burst" into 100 rows per booking instance and each booking-cluster combination provides a score.  The top 5 scoring clusters per booking are submitted as the solution.

A few details were left out in the interest of brevity as this post is already quite long, but the above more or less summarizes the ideas in play.  I'm looking forward to hearing about the approaches of the other teams.

  [1]: https://www.kaggle.com/leustagos
  [2]: https://en.wikipedia.org/wiki/Great-circle_distance#Formulas
  [3]: http://www.csie.ntu.edu.tw/~r01922136/libffm/
