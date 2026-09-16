# Simple linear regression for sigma (40th place)

Competition: ariel-data-challenge-2024
Rank: #40
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543697

In order to estimate `sigma` I decided to use the following idea:

1) You can remove a part of your spectrum (for example, wavelengths 0-25), and estimate the **s** value without the selected spectrum part (using the mean signal but you compute this mean signal excluding the selected wavelengths);

2) Estimate several values of **s** for one planet by sliding this window (I call it *dark filter* because it just removes a part of the spectrum), different windows do not overlap to make the computations faster;

3) Compute the standard deviation of **s** values;

For a single planet it may look like (notice the brown curve - it corresponds to **s** estimations with different dark windows):



4) Notice a strong correlation between std(**s**) and MSE (between true and predicted spectrum, the latter is just a constant):



5) Fit a linear or a quantile regression to predict `sigma`:

```
sigma = 10.0 * np.std(dark_filter_s) - 1.4e-5
sigma = max(sigma, 6e-5)
sigma = min(sigma, 1e-3)
```

6) Use your mean and sigma (two numbers) for one planet. Also, you can scale them at different wavelengths using the prior distribution.

P.S. Thanks to the host team for this interesting competition, and congratulations to the winners!
