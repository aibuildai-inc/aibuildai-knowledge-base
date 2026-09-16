# 5th place solution: Stack-sliding and Differential Evolution

Competition: g2net-detecting-continuous-gravitational-waves
Rank: #5
Source: https://www.kaggle.com/c/g2net-detecting-continuous-gravitational-waves/discussion/376022

We want to thank the organizers and the other participants of this great event! This is our first time on Kaggle and we are happy to win the Gold medal.

We are theoretical particle physicists working mainly on Heavy Neutral Leptons (hence the team name with the same acronym) with no background in Gravitational Wave physics. The initial plan was to improve our ML skills, but we turned eventually to a physics-based approach.



## The idea of the Method

It is easy to find a continuous signal – it is just a peak in a Fourier transform.
However, due to the Doppler modulation and spin-down of the neutron star, the signal is spread over multiple frequency bins. 
Our method (very similar to Jun Koda's solution) aims to streamline the modulation curve, as in the figure below 



(figure from this [arxiv.org/abs/2206.06447](https://arxiv.org/abs/2206.06447.))

The modulation pattern depends on the position of the source (alpha, delta) and the time derivative of the frequency F1, see the [pyFstat tutorial](https://github.com/PyFstat/PyFstat/blob/master/examples/tutorials/1_generating_signals.ipynb).

In fact, the method is not new and has been used by the GW community under the name of StaSlide [1]. Once the individual SFTs are shifted so that the signal is located in one frequency bin, we simply sum their powers (absolute values squared). If the modulation pattern mismatches the actual one slightly, the signal is spread across several bins which drastically reduces the sensitivity. Therefore, one has to scan over a very fine grid in the parameter space. The method is insensitive to any gaps in the timestamps. It is also rather robust against non-stationary noise, but very sensitive to instrumental lines.


## Implementation

For this challenge, we have implemented the method from scratch, first in python and then in Julia (the optimized Julia code gives a ~240x speed-up compared to a naive python implementation).

The processing followed these steps:
* Normalize the data. |SFT|² / std(Re(SFT)) worked just well. We tweaked this a bit when a strong instrumental line was detected by the algorithm, but it didn’t seem to affect the performance. Normalized this way, the gaussian noise will follow chi2 distribution and the signal will follow non-central chi2. 
* For every sample, scan over the parameter space (alpha, delta, f1) and find the maximum power. Scanning over alpha and delta with sufficient resolution takes ~20 s per sample, so scanning over f1 was too time-consuming for us. So we used [differential](https://github.com/robertfeldt/BlackBoxOptim.jl) [evolution](https://en.wikipedia.org/wiki/Differential_evolution) with the objective - max ( Power_L1 + Power_H1). Summing the powers from two detectors greatly improved our LB (from 0.747 to 0.804).
* During the scan, our algorithm analyzed the data to isolate potential glitches. Improving this algorithm slightly improved our score.
* Final predictions were made by simply applying the logistic function to the max Power. AUC score is invariant under reparametrization, so the parameters of the logistic function do not matter.

Processing the test set takes around 10 hours on a 5-year-old Linux desktop machine (on 8 cores).



## What could have been improved

* Amplitude modulation. The signal intensity depends on the position of the detectors compared to the source, and an extra phase in a complicated way. We wanted to sum the stacks with weights proportional to that amplitude modulation but didn’t have time to implement that properly. If we understand correctly, Jun Kodo used amplitude modulation. Simpler filters (daily/twice-daily modulation \\( \propto \exp(2πi t/T)  \\)) did not perform well.
* Maybe we concentrated too much on isolating glitches, which make up at most 2% of the test set.
* Optimal filtering. We noticed the signal leakage to the nearby frequency bins due to the finite time of short SFTs. Mitigating it with optimal filtering is a great idea, which put Jun Kodo in the 1st place. We initially tried to filter the SFTs when investigating the use of CNNs, however, we did not have time to revive this effort for stack-sliding.


## Earlier failed attempts

Like a good Kaggle beginner, we initially jumped at the most high-tech solution possible: we wanted to use a Transformer applied to time series. We realized that there is an existing method to search for CWs that generates sequential data: Viterbi tracks [2].

After this attempt failed, we then decided to temper our expectations and go for a known and tested method: convolutional neural networks (in particular we searched for noise-resilient CNNs). Here we encountered a number of problems. First, the timestamps are not nicely aligned on a grid, and the SFTs contain a large number of gaps and overlaps. The number of timesteps is also too large to feed into a typical CNN architecture. We realized that we would need to resize our input, and we tried to find a clever way to do so, that didn’t penalize our sensitivity too much (having worked previously on resonant particle searches, we were all well-aware of the importance of maintaining the best possible resolution). To this end we tried a number of filters to match the daily amplitude and frequency modulations before max-pooling the SFTs for each day.

We had limited success here: we managed to make some hidden CWs much more visible to the human eye, but when we tried to use this method to produce the CNN input, we encountered a much bigger problem: the test set was significantly out-of-distribution compared to the training set, with a number of test samples containing strong glitches (see for instance [these](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/364854#2026621) [posts](https://www.kaggle.com/competitions/g2net-detecting-continuous-gravitational-waves/discussion/364854#2022856) and [this notebook](https://www.kaggle.com/code/vslaykovsky/g2net-winning-strategy-with-external-data)). We realized that we would need to generate our own training set if we wanted to apply any machine learning method. This gave us the impetus to look at algorithmic methods that do not require a training set.

We also tried more elaborate methods of normalizing data, like the rolling median normalization, but it didn’t improve our results and seemingly increased the look-elsewhere effect. 



## References

1. LIGO Scientific Collaboration, *All-sky search for periodic gravitational waves in LIGO S4 data,* Phys.Rev.D 77 (2008) 022001, Phys.Rev.D 80 (2009) 129904 (erratum), [arXiv:0708.3818](https://arxiv.org/abs/0708.3818)
2. Joe Bayley, Graham Woan, Chris Messenger, *Generalized application of the Viterbi algorithm to searches for continuous gravitational-wave signals,* Phys.Rev.D 100 (2019) 2, 023006, [arXiv:1903.12614](https://arxiv.org/abs/1903.12614)
