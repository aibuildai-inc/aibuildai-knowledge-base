# 2nd place solution with code: refinement with FWI

Competition: waveform-inversion
Rank: #2
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587950

Link to full code: https://www.kaggle.com/code/jeroencottaar/geophysical-waveform-inversion-2nd-place

Thanks to the organizers for a fun competition, to Kaggle for providing the infrastructure, and to @brendanartley for providing an excellent starting point - I probably wouldn't even have competed without it.
## Introduction

How do you find out what's under your feet without digging? You give the ground a good thump, which sends seismic waves underground. These waves interact with and reflect off underground structures. By placing multiple detectors at the surface, you can pick up the reflected waves. This measurement is called a seismogram.

It's fairly straightforward to model the impact on the seismogram of subsurface structures, described by a velocity profile. The challenge lies in inverting this function: how do we find the original velocity profile given a seismogram? This was the challenge posed to us in this competition.


## Outline of my solution

I start with a rough deep learning model that would score 28.8 on the competition metric (this is the model @brendanartley shared a few weeks ago). This is then refined using full waveform inversion (FWI) to a score of 7.6. The challenge was making this feasible in terms of convergence and computation time. The key elements of my solution:
- Carefully designed priors for various subsets of the data.
- Choosing and tweaking of solvers (BFGS and Gauss-Newton) per subset.
- Optimizing computation speed, including custom CUDA kernels for the forward model and BFGS overhead.

Even then, my solution cost about $700 in cloud compute costs to develop and run. This is in line with other top solutions, but I'm hoping this competition is an outlier in this regard or we'll have very few competitors going for top spots soon - not to mention the environmental impact.
## Details

FWI is much easier when you start close to the solution. For this I use a deep learning model. I didn't develop this myself, but use a model @brendanartley graciously shared earlier in the competition. I modified nothing about it (and also didn't train it myself), so if you want to know more about it, you might as well get it from him: https://www.kaggle.com/code/brendanartley/caformer-full-resolution-improved

From there I perform regularized full waveform inversion. This means I'm trying to find *x* that minimizes:
$$c(x)=p(x) + ||s(x)-m||^2_2$$
Here, *x* is the velocity profile, *p(x)* is the cost function defined by the prior, *s(x)* is the seismogram that follows from *x*, and *m* is the actual measured seismogram. In words: we're trying to balance minimizing a cost function and minimizing the residual.

Typical methods to achieve this are BFGS and Gauss-Newton; I won't discuss these in detail. Both of these methods require us differentiate the cost function above; the most expensive part of this is computing the gradient of the seismogram function *\triangledown s(x)*. I heavily optimized this with custom CUDA kernels; the forward pass takes 30 ms and the backward pass 70 ms on a V100 at double precision, though faster is probably possible. I also reimplemented parts of BFGS with a CUDA kernel to reduce overhead.

What remains is the choice of prior *p(x)* and solving strategy, which took up the bulk of development time. Different priors are needed for the different families. I won't discuss how I classify them here; as many of you have noticed, this is a fairly easy problem.
##### FlatVelA and FlatVelB


The solution is restricted to 1D profiles, with total variation as the cost function:
$$ p(x)=\sum_{i=1}^{69} |x_{i+1}-x_i| $$
(Actually, it's not quite the absolute value, but smoothed around 0 to remain differentiable.)

I use a single BFGS pass, terminated when the cost function stops decreasing significantly. My public LB score for this one is well under 0.1.
##### StyleA


The prior for this one is a Gaussian Process with a squared-exponential kernel, and additional terms for noise, offset, and slope. Hyperparameters were tuned on the training set using maximum likelihood estimation. There is actually no noise in StyleA, which makes the covariance matrix *K* very ill-conditioned. I resolve this by restricting the solution to the strongest 1073 modes of an SVD decomposition of *K*.

The strategy is a BFGS pass, further refined by a single Gauss-Newton step. My public LB score for this one is around 3.4.
##### StyleB
[XXX picture]
Here I also use a Gaussian Process, with hyperparameters different from above. This one is very noisy though - it's not very far from not being regularized at all.

The strategy is a BFGS pass with 1500 iterations, followed by a second pass that is terminated based on a gradient criterion. My public LB score for StyleB is around 44.4 - I was unable to find a way to capture any of the high-frequency features.
##### Other datasets


These are again tackled with a total variation prior, but now in 2D, so the cost function penalizes absolute differences between a point and its neighbors.

The strategy is 3 BFGS passes. For the last one, I scan for areas where the velocity profile is almost flat, and restrict these to be exactly flat. Some specific datasets with only 2 flat areas are handled differently; I won't discuss this here. My public LB score for this set, which is the bulk of the data, is 4.9.

## Summary

| Dataset    | Prior                            | Strategy   | Public LB score |
| ---------- | -------------------------------- | ---------- | --------------- |
| FlatVelA+B | 1D total variation               | BFGS       | 0.0             |
| StyleA     | Gaussian Process, SVD restricted | BFGS -> GN | 3.4             |
| StyleB     | Gaussian Process                 | 2x BFGS    | 44.4            |
| Other      | 2D total variation               | 3x BFGS    | 4.9             |


## Other notes and things that didn't work

- I tried to describe the StyleB prior using a variational auto-encoder, but didn't get it to see anything but noise.
- I tried various forms of preconditioning for BFGS and GN, but they never brought any benefit.
- My forward modeling CUDA kernels are still pretty inefficient with memory. A lot more speedup might be possible by merging the timesteps (requiring a cooperative kernel), and managing caches cleverly.
- I didn't try to improve the original deep learning model; I was surprised this could achieve 28.8, and didn't think a significant improvement would be possible...
- The forward modeling involves taking the minimum of the velocity map - problematic, because this is not differentiable. I solved this by adding the minimum velocity as an additional degree of freedom (so decoupling it from the velocity profile entirely). It is never penalized in the prior.
