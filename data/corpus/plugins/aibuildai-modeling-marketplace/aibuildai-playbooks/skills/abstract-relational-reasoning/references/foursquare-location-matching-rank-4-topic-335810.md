# 4th place solution

Competition: foursquare-location-matching
Rank: #4
Source: https://www.kaggle.com/c/foursquare-location-matching/discussion/335810

Hello !

First of all, I would like to thank the organizers for this very interesting and challenging competition. Also, congratulations to all participants : the suspense was so tense in the last few days that I can’t believe it’s over now. I’m writing this at 3am, very grateful and relieved ! Also, a kind word for the deserving people who missed a medal by very few positions, because it can be hard to accept. You did really great, and I’m sure you’ll do even better the next time. Hardwork pays off.

Finally, thanks to my talented and hard-working teammates @ymatioun and @theoviel, with whom I enjoyed to work and learn.
<br>
## **<u>Overview</u>**
Our solution relies on the following steps :

•	**Clean data, mostly names.** Remove special characters + space, apply unidecode to unify non-latin languages, lower text…
•	**Create some meaningful features.** This has required some work beforehand, for instance to create groups of similar categories ('gyms', 'gyms or fitness centers', 'gymnastics gyms', 'gym pools'…), and to compute the average distance for each of them where we would be likely to find at least 50%, 75%, 90% and 95% of true matches. We also grouped names, cities and states, if they designated the same place. You can find [3 useful notebooks here](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335873) that show how we did it.
•	**Find candidate pairs to match.** To do this, we used different ideas : consider near neighbors; name-similarity; same words in names; grouped-category equality; (cleaned) phone number equality; same address; TF-IDF (especially for airports and indian places); …
•	**Pair matching.** At this point, we had so much pairs that we had to use 2 models to avoid memory errors. The first one, a 5-fold LGBM, was only using a few features such as distance or name similarity to avoid issues. Its goal was to delete every non-relevant pairs (threshold < 0.007). We then computed more than 200 features to feed a 2nd LGBM (20 folds !), which was endorsed to give the final results. Both of these models were trained on the whole train set (1,1M rows), which was not possible on Kaggle and had to be done elsewhere due to memory constraints.
•	**Post-processing.** We tried a lot of ideas (graph-based; 3rd LGBM model taking the 2nd model’s score as an input, as well as contextual features such as the number of high-scored neighbors of a places). In the end, the key idea was to adapt the thresholds to the sizes of the groups we were merging. In fact, merging 2 places has not the same impact (and probability) than merging 2 groups of 10 places.

**<u>And the magic that boosted our score from 0.939 (15th) to 0.957 (4th)</u>** : some rows of the private test set were very similar to rows of the train set. Thankfully, we found this on the very last day of the competition. So we recorded every matched ids of the train set, and if they were found in the test set, then we would automatically matched them. In practice, we made the parallel between the train and test set by identifying ids with the following key : cleaned_name + round(latitude, 5) + round(longitude, 5).
In my opinion, it would have been better for the competition if the entries of the train set and the test set were distinct.
Edit : we didn't use the leak to delete false positives  but it would have surely given a big boost.
<br>
## **<u>Diagram</u>**
 <a href="https://ibb.co/xXCkXR6">[4sq-drawio]</a>
Thanks to @theoviel for this great illustration.
<br>
**<u>What did not worked well</u>**
This part would be very long, but mainly:

•	Tf-idf was not very useful : less than 3% of true matches added, and it added some false-positives that were hard to catch (since the names were similar by definition).
•	Translation of foreign languages was not a game-changer compared to unidecode (except pykakasi that was better for Japanese translation)
•	(Offline) reverse geocoder was not that useful, mainly because the main features were foremost based on the name/latitude/longitude rather than on city/state.
•	Model stacking. We tried to add an XGBoost and Catboost, but in the end, the gain was very poor compared to the efforts.

<br>
## **<u>Code</u>**
Our code is avalaible [here](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/335921).

<br>
## **<u>Conclusion</u>**
I really enjoyed this competition, because a lot of approaches were possible. In the end, a great solution required to tackle very different and complementary tasks: text-processing, out-of-the-box ideas, models tuning, memory handling, features engineering… So everyone ends to learn something ! And thanks to it, after a lot of hard-work, I've become a Kaggle Competition Master ! ✔️

To finish, thank you all once again for your great efforts and contributions. This competition has been very fun, and I'm looking forward to continue my journey on Kaggle.

If you have any question about our work, feel free to ask, and I wish you all the best for the next competitions. 🙂
