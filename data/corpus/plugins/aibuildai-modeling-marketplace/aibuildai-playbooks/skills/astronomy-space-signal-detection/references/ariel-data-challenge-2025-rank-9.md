# 9th Solution

Competition: ariel-data-challenge-2025
Rank: #9
Source: https://www.kaggle.com/c/ariel-data-challenge-2025/writeups/9th-solution

Thanks to kaggle and everyone involved for hosting such an interesting competition. There was a lot to learn, and that's why we compete. 

**TLDR**

Our solution is a a mix of transit light curve curve fitting using either batman + scipy-opitimize or pylightcurve-torch + gpytorch. We then trained a linear NN to calibrate predictions and estimate sigma values.

We only discovered Pylightcurve-torch and GPytorch a week before the end. We then sprinted to replicate and improve the batman pipeline with these. 

**Cross validation**

Cross validation was straightforward. We simply split the star into 4 folds. We evaluated the curve fitting steps without CV as there was no training, but for the NN postprocessing we did proper CV evaluation. Overall we had very good CV / LB correlation.

**Data preprocessing**

We used @ilu000 preprocessing as a basis which has been shared here: https://www.kaggle.com/code/ilu000/ariel25-quick-data-prep-improved

We added a minor tweak for detecting cosmic rays, which was also shared in @jeroencottaar notebook https://www.kaggle.com/code/jeroencottaar/demonstrate-apparent-label-issue

We made the additional following changes. They yield over 0.005 score improvement when we implement them. 
- Inverted offset. The formula provided to use at the start of the competition was the formula used to encode the data. Given we had to decode, we have to invert both operations in the formula. Host only fixed one inversion during the competition. 
- The dark frame must be multiplied by the duration since sensor reset. We therefore updated the code this way:
```python
        # Step 4: dark current subtraction
        if sensor == "FGS1":
            dt = torch.ones(len(signal), device="cuda:0") * 0.1
            dt[1::2] += 0.3
        elif sensor == "AIRS-CH0":
            dt = torch.tensor(dt_airs).to("cuda:0")
            dt[1::2] += 0.2

```
- Jitter removal. We centered the signal by computing the center oliver spatial dimensions, then using scipy 2d interpolate to move the center to the center of the frame. Doing so we need to crop the outermost pixels.

Here is an example of correcting AIRS center on the spatial axis. Orange is after correction.

%20-%20JupyterLab.png?generation=1758906368284879&alt=media)

- Some frames have very large center deviations. This is because of cosmic rays. We simply discard these frames and replace them by the one right before it. Here is an example of two cosmic rays effect on FGS1.

%20-%20JupyterLab.png?generation=1758906523858560&alt=media)

- We did not fix dispersion changes due to Jitter. It can still be significant. Here is the same planet where we plot the normalized variance of signal in one of FGS1 spatial axis. The bump in the middle means that more signal will be cropped from a narrow square.

%20-%20JupyterLab.png?generation=1758906657017925&alt=media)

-Light is not appearing as a nice ellipsis on the FGS1 sensor. Here is the log scaled average over all planets of FGS1 signal, after centering it:

%20-%20JupyterLab.png?generation=1758906716843836&alt=media)

We masked the darker part of that average, rather than using a square crop of the center.

**Main approach**

We employed an iterative curve-fitting strategy to extract the planet-to-star radius ratio (Rp/Rs) from noisy transient light curve data. The procedure consisted of progressively refining the fit across different data representations:

1. Phase detection – The orbital phase of the signal was identified to align the transit features. We used the largest gradient average over wavelengths, as in public notebooks.
2. Baseline modeling – A degree 2 polynomial fit was applied to the out-of-transit “arms” of the light curve to model and remove baseline trends.
3. Global transit fitting – The normalized light curve was fit using the batman transit model, yielding an initial estimate of Rp.
4. FGS refinement – The fit was repeated on the FGS data
5. Wavelength binning – The same fitting procedure was applied to 18 wavelength buckets, capturing wavelength-dependent variations.
6. Wavelength binning with GP – For the bucketed data, a 2D polynomial plus Gaussian Process (GP) model was used to jointly capture systematic trends and correlated noise, refining the Rp estimate.

**Postprocessing**

1. combine predictions. We combined the results from the different curve fitting steps into 18 values for prediction and 283 values for sigma; those then go into a linear NN which weights the different sigma estimates and expands the 18 predictions back to 283 values
2. train simple NN for calibrating predictions and sigma.. We use a NN because we used a GLLoss instead of the usual MSE for linear regression.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pred_fc = nn.Linear(18, 283, bias=True)
        self.sigma_fc = nn.Linear(5, 1, bias=True)
        
    def forward(self, pred, sigma, s_std, var_l, s_std2, s_std3):
        x = torch.log(pred).float()
        x = self.pred_fc(x).squeeze(-1)

        y = torch.stack([sigma, s_std, var_l, s_std2, s_std3], 2).float()
        y = torch.log(y)
        y = self.sigma_fc(y).squeeze(-1)
        return x, y

```
