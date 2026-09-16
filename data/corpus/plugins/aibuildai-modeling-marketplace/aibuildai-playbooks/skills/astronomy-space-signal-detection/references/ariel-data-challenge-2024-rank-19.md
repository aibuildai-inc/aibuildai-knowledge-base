# 19th Place Solution - Curve Fitting

Competition: ariel-data-challenge-2024
Rank: #19
Source: https://www.kaggle.com/c/ariel-data-challenge-2024/discussion/543855

Many thanks to the Ariel team for organizing this competition ! This was my first kaggle competition and I enjoyed a lot working on this topic. 
My solution is mostly based on modelling the signal using a nonlinear function, and solving a least-squares problem using a gradient based algorithm such as the Levenberg-Marquardt algorithm.
I've just made a clean notebook of my solution here : https://www.kaggle.com/code/kangourous/ariel-19th-place-solution-curve-fitting


## Preprocessing
I used the organizer's notebook for calibration but accelerated the process by utilizing the GPU. For FGS, I calculated the mean of the 100 brightest pixels, and for AIRS-CH0, I took the 15 brightest pixels for each wavelength.

## Transit zone
I've found that an accurate estimation of the transit boundaries for an observation was crucial to get a good estimation of the transit depth. As the transit zone is the same for all wavelength, I took the mean value for AIRS along frequency axis. Initially, I used the second derivative to spot the transit zone (thanks https://www.kaggle.com/code/rezanl/code-find-transition-zone-using-derivatives), but to do so I needed to smooth the signal which led to a loss of precision. So I tried a new approach : I've created a function \\(f(x)\\) to model the signal as the product of a polynomial function \\(p(x)\\) and a step function \\(s(x)\\) :

<p align="center">
[drawing]
</p>

\\(a_i\\) are the parameters of the polynomial function, \\(z_1, z_2, z_3, z_4\\) represent the key points for the transit event, and \\(d\\) represent the transit depth.
Our least square problem consists of finding the optimal value for \\(a_i\\), \\(z_i\\) and \\(d\\) so that \\(f(x)\\) fit best the data. To achieve this, I used the `dogbox` algorithm, for which `scipy.optimize` provides an implementation. This optimization process gives 2 useful information :
1.  the transit boundaries \\(z_i\\)
2. the average background noise, modeled as a polynomial function of coefficient \\(a_i\\). I divided the signals by this curve to denoise globally the AIRS data.
We will then apply the same optimization process for each wavelength for, this time, estimate the depth.

<p float="left">
  
   
  
</p>

## Depth estimation
Now that we have the transit zone for each planet, I used \\(f(x)\\) again with the parameters \\(p_i\\) fixed, to model the signal for each wavelength. I solve the least square problem by finding the optimal \\(a_i\\) and \\(d\\) for each wavelength.  I then smoothed the result along the wavelength axis using a moving average window of size 73, weighted by a Taylor windows. This approach gave me 0.611 on LB. By analyzing sources of error, I found that my method underperformed for planets with spectra that have high standard deviation. For these planet, the moving average windows was to aggressive and flatten the spectrum. So I used a smaller window size if the standard deviation of the predicted sprectrum is higher than a threshold. I got an improvement of 0.03 on LB with this adjustment.



## Sigma estimation
I only used the global root mean squared error for all the dataset.

## Improvement
During the whole competition I tried to analyze each pixel individually, to see if it was possible to denoise, or find the transit depth at a pixel-level resolution. The thing is, it is very hard to understand what happen when you visualize the signal for a single pixel 😅. There is a lot of noise and this mysterious spike that happens everywhere. I noticed that the position of the spike was consistent across each pixels for a given planet, and there is positive and negative spikes so when we take the mean they compensate each other. I was able to model the spike using a  lorentzian function, but I did nothing with that because denoise each pixel individually was to heavy, and it seems that it didn't improve my results.

<p float="left">
  
   
</p>
