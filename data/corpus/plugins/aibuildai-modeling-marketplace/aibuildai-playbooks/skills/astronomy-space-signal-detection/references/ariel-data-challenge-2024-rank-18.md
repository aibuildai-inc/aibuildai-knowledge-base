# 18th Place Solution

Competition: ariel-data-challenge-2024
Rank: #18
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543675

# 18th Place Solution

This competition was incredibly engaging, with active discussions, including those initiated by the hosts. I want to express my gratitude to the organizers for providing such a wonderful opportunity.


## Solution Summary



My approach is based on [an excellent public notebook](https://www.kaggle.com/code/vitalykudelya/neurips-ariel-data-correlation-parallel-scale). I would like to thank the author of that notebook and the many authors of other notebooks that served as its foundation.

My solution is a straightforward extension of the public notebook. The main enhancements are as follows:

- I modified the polynomial fitting of the flux averated in frequency direction, which was previously done in two dimensions, to fit a three-dimensional plane, including the frequency direction.
- I adjusted a constant sigma to be estimated individually for each planet.


Regarding the three-dimensional plane fitting, there are several considerations due to the high noise levels in individual frequency signals:

- Normalize each frequency individually by its average, as signal intensity varies by frequency.
- Use the transit phase calculated at the mean frequency directly.
- Pre-adjust the signals to simulate a depth-free condition using the depth calculated at the mean frequency.
- Instead of optimizing the depth during three-dimensional plane fitting, normalize flux using the fitted signal and estimate the depth from the ratio of the average flux in transit phase signal to other signals.

## Does Not Work For Me
The signals at frequency indices 180 to 186 were particularly puzzling. Despite having significantly larger errors than other frequencies, using these signals to predict all frequencies with Linear Regression led to an impressive RMSE improvement across many frequency bands (with the best CV score reaching 49 ppm). However, this model performed poorly on the test data, failing to deliver any meaningful results.
