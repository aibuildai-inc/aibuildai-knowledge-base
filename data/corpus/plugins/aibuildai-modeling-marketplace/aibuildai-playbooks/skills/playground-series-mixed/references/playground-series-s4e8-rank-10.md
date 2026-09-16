# 10th place solution (and a potential 5th)

Competition: playground-series-s4e8
Rank: #10
Source: https://www.kaggle.com/c/playground-series-s4e8/discussion/531424

Firstly, congratulations to @optimistix, @neupane9sujal, and AutoML Grandmasters for their impressive results in this competition. Kudos to @optimistix for staying at the top of the leaderboard for most of the competition and securing their first 1st place in the Playground Series. Well deserved, @optimistix! I can't wait to read about your winning solution.

### (First) Final Solution 
I submitted my (first) final solution on Friday. It was an ensemble consisting of 9 of my best-scoring individual models. Although this solution had a low public LB score, I was happy with it, mostly because at that time, I didn’t have any ideas on how I could improve it further. I was just crossing my fingers for a big shakeup at the end!

The table below shows the models in this ensemble along with their 5-fold CV and public LB scores.

| Model               | **5-Fold CV** | **Public LB** |
|:--------------------|:-------------:|:-------------:|
| AutoGluon           | 0.98492       | 0.98523       |
| XGBoost (dart)      | 0.98490       | 0.98499       |
| XGBoost             | 0.98488       | 0.98503       |
| LightGBM (dart)     | 0.98482       | 0.98507       |
| LightGBM            | 0.98480       | 0.98501       |
| HistGB              | 0.98474       | 0.98496       |
| XGBoost (rf)        | 0.98465       | 0.98473       |
| CatBoost            | 0.98454       | 0.98487       |
| Neural Network      | 0.98434       | 0.98469       |
| **Ensemble**        | **0.98501**   | **0.98521**   |

## One Last Experiment 
Yesterday, just a few hours before the competition ended, I decided to run one last experiment. I collected the OOF predictions from all my experiments throughout the competition and ensembled them. I ended up with 32 models, including the original 9. This increased my CV score to 0.985050 and my public LB score to 0.98528.

I discovered that some models had the same score in every fold, even though their OOF predictions weren't exactly the same. After removing these models, my CV score increased to 0.985053 and my public LB score to 0.98531. So, I decided to select this and my original submission from Friday as my final two submissions.

Today, I found that my 32-model ensemble, which probably contained some duplicates or very similar models, had the highest score on the private LB and could have secured me 5th place. Ensembling very similar or duplicate models goes against my intuition, which is why I didn't have multiple models of the same type in my original ensemble. I probably would never have chosen this ensemble for my final submissions due to the (likely) duplicate models, nor will I in the future. I think it was just pure luck that this ensemble received a higher score on the private LB compared to my other ensembles.


### 10th Place Solution 
Anyway, in my 10th place solution, I used 28 models and ensembled them using logistic regression. The inputs to logistic regression were converted to logits before training. I tried various functions on the inputs, but logits provided the best CV score, so I decided to go with that.

Here are the 5-fold CV scores of all models in the ensemble:


One interesting thing I noticed today is that tuning the threshold helped increase both the CV and public LB scores, but not the private LB score. I tuned the threshold using Optuna and OOF predictions, but I also tried using `TunedThresholdClassifierCV`. However, the latter didn’t improve my scores, neither the CV nor the public LB score.

---

Lastly, I would like to thank @ambrosm, @siukeitin, @omidbaghchehsaraei, @rzatemizel, and @oscarm524 for their code and discussion posts. I have drawn inspiration from and learned a lot from their contributions.
