# 7th place (public LB) solution : NN + Survival LGBM

Competition: nfl-big-data-bowl-2020
Rank: #6
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119433

First a big thanks to the organizers for this very interesting competition. The task is challenging and the players tracking data is so fascinating.
Here is a summary of our final solution. It is a mix between automatic features learning and manual features engineering. 
The final solution is a blend of 1 deep learning model and 1 LGBM. The DL model and write-up is mostly my part while Sebastien worked on the Survival modelling with LGBM. 

Thanks @sebastienconort it was a pleasure to work together on this competition.

### Validation
We used last 2000 play_id as validation. At the beginning is has a stable gap of 0.0001 with LB but in the end we started to overfit the validation. A posteriori, we may have treated this too lightly and some other teams have share better validation scheme.

### Survival modelling with LGBM for learning “Yard” distribution -  by @sebastienconort 
In this part, unlike many competitors, we did not approach the problem as a multi-class problem, but as a regression problem, with a survival modelling approach on the number of Yards reached. The “time“ variable is the number of “yards run so far” and the “death” event is the rusher being stopped. 
If you are interested for more details, see the note at the end o this post: (Notes on Survival Modelling for Yard prediction)
Depending on time and feedback from readers of this post, we might publish a tutorial notebook describing this approach which is very useful for churn prediction, or insurance pricing 😄 
Without the NN embeddings descrived hereunder, and with descriptive variables on Defense and Offense players ordered by their distance to the Rusher, this LGBM approach reached 0.0131 on the public LB. 

### NN Architecture 
 The LGBM takes as inputs some intermediate embedding of the NN with other variables. 

The deep learning architecture has differents parts:
* An MLP that generate an interaction embedding for each couple (player_i, player_j).
* These interaction embeddings is summed for each player_i then concat with other inputs for players_i. This is then fed into another MLP to create players_embedding
* The sum of all players embedding is concat with play-level  features before feeding to final classifier.

Globally the interaction part is like a Graph NN with 1 round of message passing.
Separating player features, context features, and the terrain limits features (YardLine and DistancetoYardline) and treat them separately make it easier to regularize the network.

### Features
* Most values are from players positions and speed, we found that play-level variables helped a little. 
* We projected positions in 0.5s, 1s, 1.5s with the Speed and these were the best features for LGBM. However we did not see improvement when doing this with NN
We did not see improvement when adding voronoi features on top of distances.
* For NN, we found big gain to include in the interaction features the relative position of player_j projected on the direction from player_i to the rusher. 

###Deployment
One of the challenge of the competition is the deployment. We did modify our code to be quite conservative in the end. We also made the  choice to paste the weights in the kernels and refinetune the model from that weight. 
LGBM is always retrain since it is fast (~1 min)
Finally for the 2 submissions, we chose 1 with NN finetuning and 1 without

### Tricks that helped
* For deep learning, we fit a blend of 10 best parameters found by random search then distill the output to 1 DL model. The weight of this final model is pasted in the solution.
* We made a mistake as some point to do batchnorm on inputs. Which means measures in X and Y axes are scaled differently. This make learning very unstable. Removing this and scale inputs manually helped.
* Some tricks that helped a little bit each time: 
  * augment data by flipping Y-axis
  * add noise to target in training
  * add a l1 penalization in the output of consecutive bins of the softmax  (fused lasso)
  * Rotate Orientation by 90° in 2017

### What did not helped
* We tried Transformer many times but it did not do better than our architecture. Maybe we should try again after reading other solutions :)
* We tried to add more layer of interactions between players but it did not seems to help either
* We tested different activation like ELU and SELU but ReLU does much better.
* Normalizing S and A in 2017 did not work for us. We did not think of treating these as missing values like https://www.kaggle.com/wimwim. Huge regret and big kudos for him with this nice data prep work.
* We tried to predict the real distance travelled by the rusher instead since it is more natural in our pov. However, to our surprise it is worse, our hypothesis is that because the defense is organized around the yardline more than around the runner ? But we are interested to hear experts opinion about this.

### Notes on Survival Modelling for Yard prediction
Survival modelling consists in learning the distributions of a “death” event while dealing with potential censorship or truncature in the observation of that event: either the Survival Function (= 1 – Cumulative Probability Distribution) is learned, or the Cumulative Hazard Function, or the Hazard Function. Knowing one of these functions enables to know all the others, as there is 1 to 1 mapping. In this challenge, there was neither truncature nor censorship, but we were eager to test this survival approach anyway.
A nice generic approach to learning the Hazard Function is by assuming it is piecewise constant (this approach is referred to in literature as Piece-Wise Exponential Model, and is implemented for instance for linear models in the Lifelines python library). Here we assumed the constant step of our piecewise constant function was 1 yard. This approach enabled us to use a LGBM regressor with a Poisson objective. Indeed in the case of the Piece-Wise Exponential Model, the negative log-likelihood of the survival problem is equal (+ a constant term not important for the loss minimization) to the negative log-likelihood of a Poisson regression, the so-called “Poisson trick” as explained in chapter 7.4 of these Princeton class notes (https://data.princeton.edu/wws509/notes/c7.pdf )
To be able to apply this approach, the data set has first to be extended: each line is duplicated in multiple lines from -99 to “Yard” which is the death event. Then two new columns are added in the dataset : a descriptive variable being “yards run so far” taking values from -99 to “Yard”, and a new target (let’s name it “Stop” variable) being equal to 0 everywhere except at lines where “yards run so far” == “Yard” (death event). The “yards run so far” variable is very important to be included in the regression, for the LGBM to learn the shape of the Hazard Function.
The disadvantage of this approach is that it increases RAM consumption because of data duplication. But it has the advantage of having only one regression target. As in the train set the minimum observed value for “Yard” is -14, we suppressed all lines with ‘yards run so far” equal from -99 to -15, and extrapolated manually with a 0 value our Hazard Function in post processing.
Once our LGBM is trained, it is able to predict the Hazard Function for different “yards run so far” values, which gives the probability of the rusher being stopped knowing he has not been stopped so far. 
To compute the Cumulative Distribution Function, we predict the Hazard Function for values of “yards run so far”  from -99 to 99, and we postprocess the resulting numpy array “HF” with the following formula : 1 – np.exp (- np.cumsum(HF)).
