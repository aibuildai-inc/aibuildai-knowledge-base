# 19th Place Solution

Competition: ariel-data-challenge-2025
Rank: #19
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/19th-place-solution

Thanks very much to the organisers, and my fellow competitors for an interesting and challenging competition.  I really appreciate these challenges, even when they're not directly in my field of research.  The cross disciplinary problem solving, and more general lessons are invaluable to me.

My final solution was a mixture of careful curve fitting, binning in both wavelength and time, then interpolation in wavelength and using nested LGBM models to improve the fitted estimates, and also to estimate the uncertainty.

The full training and inference notebook is on GitHub  [here](https://github.com/Wologman/Kaggle_Ariel_Data_Challenge_25/blob/main/ariel25-batman-lgbm.ipynb)

The first stage of the curve fitting is shown below.  Here I fitted a 'baseline' curve to a time-binned and wavelength-averaged signal for each planet.  I relied on only one breakpoint.  Both the out-of-transit part, and the fully occluded part share the same linear term.  I figured the fit didn't need to be perfect at this stage, just consistent, and handle edge cases gracefully, as I would later try to correct for any shortcomings with my boosting models.



I then used the centres, breakpoints and linear correction term to produce 13 more wavelength-specific fits.  These were later interpolated to fill out all 283 predictions.  I used the features from these 13 fits to produce an LGBM based correction, and a second LGBM model with a nested data split to estimate the uncertainty.    I also used the [batman model](https://lkreidberg.github.io/batman/docs/html/index.html) for curve fit only to generate a additional features for the LGBM model.

 

Thanks again, and I hope to see everyone next time.
