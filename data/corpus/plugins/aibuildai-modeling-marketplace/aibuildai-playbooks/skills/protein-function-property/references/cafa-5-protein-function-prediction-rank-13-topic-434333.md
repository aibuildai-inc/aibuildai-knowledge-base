# A brief overview of 13th-place LB solution

Competition: cafa-5-protein-function-prediction
Rank: #13
Source: https://www.kaggle.com/c/cafa-5-protein-function-prediction/discussion/434333

It was an incredible adventure and the first competition I ever took part in! I was very lucky to find an experienced teammate - @ogurtsov , who was the leader of our team.  This journey was not just about algorithms and data; it was a journey of exploration, curiosity, and learning, so thanks to the organizers, fellow participants, and the Kaggle community!

Here is a brief overview of our 13th-place LB solution:

**1) Data:**
We used the initial train set and also annotations from the UniProt and QuickGO databases (filtered terms with NOT qualifier, evidence code "ND", obsolete terms, and those that are not present in the competition OBO file) in two ways: to update the train set (proteins with experimentally confirmed terms only) and to merge with predictions (test proteins with terms never experimentally confirmed).

**2) Features**

ESM2 3B protein sequence embeddings kindly shared by @andreylalaley with Z-score normalization (means and SDs of train data applied to train, validation, and test).

**3) Targets:**
Our target selection involved selecting the top 500 GO terms based on their frequency, or terms with a frequency above a certain threshold (BP > 250, MF and CC > 50).

**4) Neural Networks:**

The modeling strategy involved multiple Multi-Layer Perceptrons (MLPs) with different hyperparameters and layers, written with Torch or Tensorflow, R language. Each was trained on the initial or updated training set and different sets of targets. To ensure robustness, we employed cross-validation (CV) with 10 folds (one for validation and 9 for training) followed by averaging the predictions. For some models, we found it more effective to validate on train proteins only, in these cases, we used randomized 10-fold CV to avoid reducing the validation set. Importantly, we trained a separate MLP for each ontology, each time only on these proteins that have at least one term of the selected top.

UPD: For all MLPs we used AdamW optimizer,  [once cycle learning rate](https://torch.mlverse.org/docs/reference/lr_one_cycle.html) and early stopping callback (max 100 epoch, monitored metric - valid loss, with patience 1, 3 or 5) or [callback reducing the learning rate](https://rdrr.io/cran/keras/man/callback_reduce_lr_on_plateau.html) by a factor of 0.5 (patience = 1)

**5) Post-processing:**
First, we filtered the predictions using taxon constraints to eliminate terms incompatible with certain species. Also, after each step, we made corrections to ensure that the predictions of parent terms were not smaller than the predictions of child terms (in such cases we propagated max prediction of child terms). Finally, before the final merge with GO annotations, the predictions were filtered again to include only terms in the top 450 by frequency based on the original training set and also rare terms if they had predictions > 0.5.

**6) Ensemble:**
Predictions from 6 MLPs were simply averaged. The resulting predictions were averaged again with exported GO annotations (the set of test proteins with terms never experimentally confirmed).

For the secondary solution, we did weighted averaging with GO annotations and selected the top 35 predictions for each ontology.

I hope that this will be beneficial for others, serving as an example of what can be done (if we shake up) or what should be avoided (if we shake down at the end) 😆.
