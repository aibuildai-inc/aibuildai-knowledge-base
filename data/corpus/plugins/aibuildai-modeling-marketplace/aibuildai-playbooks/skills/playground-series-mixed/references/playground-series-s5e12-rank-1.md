# 1st Place Solution | Hill Climbing + Ridge Ensemble

Competition: playground-series-s5e12
Rank: #1
Source: https://www.kaggle.com/c/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl

Happy New Year, Kaggle Community! 🎉

This win came somewhat unexpectedly, but it validates the sustainability of my favorite Hill Climbing (HC) approach I've been refining over multiple previous competitions, consistently achieving Top 1% places. This was an incredible learning experience, and I want to thank everyone who shared their insights throughout the competition – especially @masayakawamata for the shared post-cutoff cross-validation approach, and @daylighth, @laureanoarcanio, @tilii, @mikhailnaumov, and @siukeitin for their discussion posts and scripts.

This 1st place finish was particularly satisfying because it came after a frustrating plateau. My HC ensemble was stuck at post-cutoff CV 0.7088X and Public LB 0.70722. Any addition of new models caused improvement in CV, but Public LB dropped. I decided to apply Ridge Ensemble Stacking using the top models from HC selection as input. I experimented with different alphas on rank-transformed predictions and different numbers of models. A plateau was achieved here as well, with post-cutoff CV of 0.70860 and Public LB 0.70739. I stopped at this stage as the gap between post-cutoff CV and Public LB of 0.00121 seemed acceptable.

I suppose that the good final result was ensured through leveraging many diverse base models (tree-based and neural networks) with different hyperparameters (depth, learning rate, regularization), feature engineering (ratios, polynomials, target encoding), seeds (multiple runs for variance reduction). I included in HC both single models and other ensembles. Such a mixed approach worked for me in previous competitions.

In the Ridge Ensemble, I focused purely on performance (post-cutoff AUC) rather than correlation-based diversity. The diversity came naturally from HC's exploration process, which tends to select models that complement each other.

The Ridge Ensemble (based on Top-36 models from HC and alpha = 10) had slightly lower post-cutoff CV than HC (0.70860 vs 0.70886), but performed better on both Public and Private LB. This demonstrates that stabilization from Ridge regularization sometimes can be more valuable than raw CV optimization.

With Ridge Ensemble based on Top-34 models and alpha=5, it was possible to achieve a Private LB of 0.70514, but the Public LB score of this ensemble was too low (0.70734) to confidently select it for final submission based on the gap.



Key takeaways in this competition were:
1)	distribution shifts are competition killers - if CV improves but LB drops, it is necessary to investigate data structure; 
2)	sometimes a slightly lower CV with better generalization wins; 
3)	two-stage ensembles may improve final results when single-stage methods reach plateau; 
4)	test multiple configurations;
5)	always trust your CV and trust the process. 

Wishing everyone success in upcoming Kaggle competitions, exciting opportunities in real-world ML projects and continuous learning and growth in data science.
