# Our (frustrated) approach

Competition: dfl-bundesliga-data-shootout
Rank: #34
Source: https://www.kaggle.com/c/dfl-bundesliga-data-shootout/discussion/359855

Although it is the norm to share successful solutions, our team thought that our (frustrated) approach could be interesting and thus should be shared.

From the beginning, we thought there was a lot of useless noise in the videos that may prevent the model from learning the right features. Based on this, we put a great effort into the preprocessing side, trying to extract only what we considered essential information. The result is the following:

| original | preprocessed |
| :---: | :---: |
| [[original]](https://www.youtube.com/watch?v=GlUgM2TUPCw) | [[preprocessed]](https://www.youtube.com/watch?v=VoAcptNeF1k) |

In summary, as can be seen in the video, we did the following:

* Detect the lines of the football field, via a Top-hat filter.
* Detect the players and the ball, via a fine-tuned YOLOv7 first and a custom-trained RetinaNet after doubts about the GPL license arouse.
* Detect each player's team using K-means with the color of the player's jerseys.
* Detect the skeletons of the five players closer to the ball, via mmpose.
* We rendered all this information together over a black background, hoping to aid the model in the learning process. We even cropped each frame centered in the ball to further reduce the noise (the result is available in [this video](https://www.youtube.com/watch?v=IZjJMa4u-Ss)).

It was a great challenge to put all these pieces together. Moreover, it was even harder to make the pipeline efficient enough to comply with the time constraints. Nonetheless, after a great effort by the team to optimize the source code of the libraries we used (specially mmpose), we were able to make the preprocessing pipeline run in half of the video duration. This took a great amount of time, but we were very happy with the result. At least until we started training and validating the model... 😅

The odyssey began when, no matter which architecture we tried, we could not get reasonable results. General action recognition models (SwinTransformer3D, SlowFast, ...), soccer-specific models (NetVLAD++, ...), custom architectures based on temporal information, ... We tried them all, but none of them worked. Even with the original videos (without preprocessing), we could not get a good result. Even after augmenting the training data with the annotated videos of SoccerNet-v2, the boost was not enough.

We have checked and rechecked for bugs, but we could not find any issues. We are very curious to see what top teams tried and check if we missed something critical.

Although the outcome was not the one we expected (we are hungry for gold to become Kaggle Masters), it was a great learning experience for the team. Thanks @alejandrobravoserna @miguelgonzalez2 @danielguzmanolivares @alvarozaera for the effort! 🥇will come ...

If you have any doubts about our approach, do not hesitate to ask!
