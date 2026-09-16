# 5th place Solution

Competition: otto-recommender-system
Rank: #5
Source: https://www.kaggle.com/c/otto-recommender-system/discussion/382802

Firstly thanks to everyone for sharing so much on this problem. I learned a lot from all of you.

Espsecially thanking (in alphabetical order)

1. Carno: For sharing your numba pipeline. All of my candidate generation and many of my features were created using numba. I had almost never used numba before, so learnt a lot of numba in this comp.

2. Chris: For sharing so much from comp start to end. I think every competitor owes you for that. I think you answered questions about the ranking model till the last day of the competition, so nothing but respect.

3. Radek: For introducing us to polars, the speed of polars while joining tables really helped speed up my experiments

4. Senkin: For your 1st place solution in H & M. Many of my ideas in otto were inspired by this.

My best performing solution on the private LB was almost single model (scoring almost the same as the ensemble), so will describe the single model (public LB: 0.604 and private 0.603)

**Candiates Genreration(Numba) -> Feature Creation (Numba , Polars) -> Ranking  Model (Lightgbm) -> Inference (Treelite)**

#### Candidates Generation

I think having a strong candidate generation method helped me a lot, so here's how I did it.

**Number of Candidates Per Session:** I generated 80 candidates most of the comp, and then jumped it up to 120 cands at the last week for some score boost (of around 0.001). I had a really decent max recall of 0.648 (on the validation set) for 80 candidates. I also tried 200 candidates in one experiment, but that did not help with the score.

Also if I take the first 20 candidates from my candidates generation model, my score on LB would be 0.585. Th

For candidates generation I used something similar to covisit matrices, I divided the user actions in a session for any given 2 aids, into various categories like

a. Any action to Any action
b. Click to cart
c. Cart to order
d. Order to order,
... etc

To keep memory usage low, I chose only the top (k * 100) candidates. K here is the number of candidates I wanted to generate

Also, I normalized the weight by the frequency of the first item. So thinking its  basically like out of 100 times that milk was purchased, how many times were eggs purchased with it.

The weight in the matrices was normalized by the number of items visited between the 2 aids we are talking about.

Let's say if we have 5 aids, aids1, aids2,  aids3, aids4, aids5

Then the weight of (aid1, aid5) will be (5-1)/(frequency of aid1).

Also weight of (aid5, aid1) will be wt of ((aid1, aid5))/2 (just to add something like purchase of aid5 was driven by purchase of aid1 and not the other way round)

For the inference time, to decide which top k candidates should I take, I used optuna, keeping the things like weight of each covisit matrix, weight of the recency of the item, normalized overall frequency of the item  etc as a parameter.

#### Feature Generation

* Basic Features like frequency of the item, clicks to carts ratio, etc, recency of the item visited (this helps a lot if number of candidates in session is more than 20).
* Association of a generated candidate to any already seen aid in the session. This could be created by using the covisit matrix weights. Going really deep into such features helped me boost my score a lot. The idea is covisit matrix could be created in different ways to establish the relationship between 2 items, for example:

  a. Take only the average distance (number of aids between) between 2 aids.
  b. Distance could also be measured in timestamp difference.
  c. Consider only candidates in  the 1st neighbourhood (immediate candidates).
  d.  Consider only relationships in the last week, etc.


#### Training a Ranking Model:

I used lightgbm, with 5% negative sampling and around 400 features, and data for last 2 weeks.
Adding data for second last week boosted the score by about 0.0005.

Some things or tricks that worked for me:

1. Training all clicks, carts, and orders with a single model (not 3 separate models)., this locally was easily seen to be getting around 0.001 to 0.002 better score (data is grouped into sessions not into session, type).

2. Using separate labels for clicks, carts and orders whiel  ranking, with ranking label gain as Orders (6) -> Carts(3) -> Clicks(1), instead of just using 1 when the user performed an action(click, carts and orders) and 0 when they did not (this boosted the score by around 0.0005).

3. Using the ranking of the stage 1 candidate generation model, as a feature of the stage 2 model. If you think, the stage 1 model can score 0.585 on the lb, so using this ranking was the important feature of my model.

#### Inference:
Nothing much to say here, except that I  used treelite for inferencing to reduce the inferencing time.

And finally congratulations to all the winners ! It was really fun participating.

P.S: I wrote this in a hurry before starting my office work, so let me know if I messed up some details
