# Repeated evaluation and variance control

Read in step 12 when scores are noisy.

Under random sampling, a small math-contest evaluation has large variance.

You must tell apart:

- the checkpoint is truly stronger;
- sampling happened to draw the correct reasoning;
- some problems flip in a stable way across several models;
- a temperature change shifted the distribution.

You can compute:

$$
\hat p=\frac{\sum_{r=1}^R\sum_{i=1}^N \mathbf 1[\text{correct}_{r,i}]}{RN}
$$

and per problem:

- always correct;
- always wrong;
- flaky.

## Watch the method risk

The aggregate benchmark score is a safe selection signal. This loop is not:

1. look at the specific test failures;
2. build training data from those failed problems;
3. go back to the test set to check;
4. repeat until the test problems are covered.

Each turn of that loop moves test content into the corpus, so the number it produces no longer predicts anything on unseen items. Variance control is pointless once the measurement is fitted.
