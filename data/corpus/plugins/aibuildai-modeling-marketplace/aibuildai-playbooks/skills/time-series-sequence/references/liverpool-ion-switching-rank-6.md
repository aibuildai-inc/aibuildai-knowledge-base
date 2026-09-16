# 6th place solution, Factorial HMM with a shared factor

Competition: liverpool-ion-switching
Rank: #6
Source: https://www.kaggle.com/c/liverpool-ion-switching/discussion/153932

Our model is one of factorial hidden Markov models (FHMM). See this paper for details of FHMM. 
http://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/factorial-HMM-97.pdf

The FHMM for this completion is illustrated in the figure bellow.  


 \\( M\\) and  \\( N\\) represent the number of channels and time steps respectively.   \\( Z\_{mn} \in \\{0,1,2,3\\} \\) represents the hidden state of channel  \\( m\\) at time  \\( n\\).  The hidden state of each channel follows the 1st order Markov process independently with the shared transition probabilities; \\(  P(Z\_{m,n+1}=k | Z\_{mn}=j) =A\_{jk} \\). \\( C\_n\\) is the number of open channels at time \\(  n \\) and it is calculated as  \\( C\_n = \sum\_{m=1}^M F(Z\_{mn}) \\), where \\( F(0)=F(1)=0\\) (close states) and  \\(F(2)=F(3)=1\\) (open states).  \\( S\_n\\) represents signal at  time \\( n\\) and it is given by Gaussian emission; \\( P(S\_n| C\_n) = \frac{1}{\sqrt{2\pi}\sigma}  \exp \left[ -\frac{1}{2\sigma^2} (S\_n-aC\_n-b)^2\right] \\). 

In our model above the adjustable parameters are \\(A,a,b~{\rm and}~\sigma \\). 
In the training phase \\( A \\) is optimized, and the other parameters are optimized in the prediction phase. 


(1) Training phase; obtain A   
Using the time-series of open_channels in the train data, the transition probability A is optimized to maximize the likelihood function  \\( P(C)=\sum\_{Z} P(C,Z)  \\), where \\( C=(C\_1,C\_2,...,C\_N) \\) and \\( Z =(Z\_{11},Z\_{21},...,Z\_{M1},Z\_{12},Z\_{22},...,Z\_{M2},...,Z\_{1N},Z\_{2N},...,Z\_{MN}) \\). The usual EM algorithm is used for this maximization. The computational cost is \\( O({}\_{M+3} C\_{3} N) \\) for one step of the EM algorithm.  This cost is largely  reduced from the cost \\( O(4^{M+1} N) \\) of the original FHMM in the above paper.  
Below notebook shows the implementation for this phase.  
https://www.kaggle.com/shimizumasaki/6th-place-solution-training-phase

(2) Prediction phase; obtain \\( a,b,\\) and \\( \sigma \\)   
In the test data predictions of the number of "open_channels" at each time point, \\( C\_{n} \\), are calculated as bellow. Using the time-series of "signal"  \\( S \\) in the test data and the transition probability obtained in the phase (1), the remaining parameters, \\( a,b,\sigma\\), are optimized to maximize the likelihood function  \\( P(S)=\sum\_{C,Z} P(S,C,Z)  \\). This optimization is basically same as in (1).  The posterior probability of \\(C\_n\\) under given signal, \\( P(C\_n|S) \\), can be obtained at the end of the EM algorithm for this optimization. Then, predictions of "open_channels" are decided from MAP estimation, \\(Pr\_n={\rm argmax}\_{C\_n}  P(C\_n|S) \\).   
NOTE: We fixed \\( a \simeq 1.234  \\) since the EM algorithm fails to obtain it in the sections with the lowest open activities.  

https://www.kaggle.com/shimizumasaki/6th-place-solution-fhmm-with-a-shared-factor

(3) noise removal   
Using the predicted time-series of "open_channles",  \\(Pr \\),  the time-series \\(S - aPr -b \\) is decomposed into Fourier series by short-time Fourier transform. Then, several peaks of power spectrum are removed, and the inverse short-time Fourier transform is used to obtain clean signal. Phases (2) and (3) are repeated several times to clean signal gradually. See above notebook for details of our noise cleaning. However, we don't think our cleaning process is good way.
