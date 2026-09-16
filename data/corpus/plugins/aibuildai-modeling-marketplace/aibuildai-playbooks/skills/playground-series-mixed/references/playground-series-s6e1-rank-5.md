# 5th place - Feature-Driven Diversity & Iterative Ensembling

Competition: playground-series-s6e1
Rank: #5
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/5th-place-feature-driven-diversity-and-iterative-e

First, I want to express my deepest gratitude to the community. This journey was built on the shoulders of giants:

- @cdeotte: For the teaching and the "Gold Standard" notebooks.

- @masayakawamata: Who taught me how to organize my experiments and code much more professionally.

- @tilii7: For the invaluable lessons on ensembling and the importance of using GP and AE.
 His "[For the (recovering) blind blending addicts] (https://www.kaggle.com/competitions/playground-series-s5e12/discussion/651787)" advice was my north star.

- Humberto Brandão: For his generous guidance and the exchange of tips , which helped me refine my approach ([Building AI Model to Detect Structural Breaks in Real Markets] (https://www.youtube.com/watch?v=2g0nIHYIxyo)).

- @mahoganybuttstrings: Who immediately recognized the strength of my results and encouraged me to keep pushing. His shared work and insights are of great value.


###  1. The Strategy: Diversity Through Engineered "Views"
As @mahoganybuttstrings [noted] (https://www.kaggle.com/competitions/playground-series-s6e1/writeups/1st-place-ive-ran-out-of-catchy-phrases-v), my ensemble showed unusual diversity. This was achieved through "Feature-Driven Diversity"—creating distinct versions of the dataset to provide different "views" of the signal for my models.

- The Hybrid Mix: I blended traditional feature engineering with Autoencoders (AE) for latent representation, Genetic Programming (GP) for symbolic relationships, and PCA for decorrelation.

- Mixing Logic (Diversity over Selection): Instead of spending time on exhaustive hyperparameter tuning or aggressive feature selection, I focused on extreme subset diversity. My "feature selection" was minimal: I ran an initial XGBoost and pruned only the features with zero importance to save time. I then fed different subsets of these features to base learners using single seeds and 5-fold CV.


### 2. Iterative Development & Stacking
My development process was strictly driven by the results of my Ridge ensemble. Every time I reached a performance plateau, I pivoted to a new technique—implementing strategies like those shared by @cdeotte—and combined the resulting features in different ways to see if they provided a non-correlated boost to the stack.

### 3. Genetic Programming Insights
One of the most effective "views" came from Genetic Programming. While no single feature was "magic," they added critical diversity. Here are the symbolic formulas that appeared most frequently:
- Environment Impact: $$StudyHours + FacilityRating + \sqrt{ClassAttendance}$$
- Study-Attendance Ratio: $$StudyHours + \sqrt{ClassAttendance}$$
- High Effort Score: $$(2 \times StudyHours) + \sqrt{ClassAttendance}$$
- Tech-Wellness Proxy: $$StudyHours + \sqrt{(InternetAccess - SleepQuality)_{clipped} + 1}$$

#### 4. Impact of Collaboration (RealMLP Integration)

Integrating the [shared OOF and Test predictions] (https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e1-realmlp-cv-8-58748-lb-8-58006) from @mahoganybuttstrings provided a massive and perfectly coherent boost across all metrics. This confirmed the robustness of the architecture when combined with my feature-driven ensemble.

The improvements were as follows:

- CV: 8.58053 → 8.57613

- Public LB: 8.53788 → 8.53309

- Private LB: 8.57875 → 8.57353

### 5. Final Results
My final solution was a Ridge stack built exclusively on [my own generated OOFs] (https://www.kaggle.com/datasets/mirko45/s6e1-results), avoiding public predictions to ensure the CV remained a true reflection of my feature work.
