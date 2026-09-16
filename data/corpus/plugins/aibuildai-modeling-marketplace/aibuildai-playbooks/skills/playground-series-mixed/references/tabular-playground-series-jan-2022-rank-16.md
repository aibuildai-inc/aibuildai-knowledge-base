# 16th place - Hybrid model

Competition: tabular-playground-series-jan-2022
Rank: #16
Source: https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/304413

I woke up this morning to find that I had jumped 277 places on the leaderboard and finished in the top 2%! Considering my time series knowledge before this TPS was precisely 0 I am a bit shocked.

You can find my full notebook [HERE](https://www.kaggle.com/samuelcortinhas/tps-jan-22-quick-eda-hybrid-model) so I won't repeat everything I did as most of the ideas are from other people. 

A few ideas of my own that I didn't see other people try though:
* **Fourier features of order 1** - to paraphrase a saying from @teckmengwong's profile:  *"If it walks like a sine wave and quacks like a sine wave, then you can treat it as if it were a sine wave."* 
* **Hybrid model + grid search** - a hybrid model was very well suited for this dataset because of the strong seasonality pattern. I added a grid search function to fine tune the parameters of my models and then ensembled these together. 
* **GDP per capita** - I was excited when this improved my public LB score but when I looked more into it, ordinary GDP was (slightly) higher correlated to the target feature. I ended up not using GDP_PC in the end but it was fun to try.

I just want to end by thanking the kaggle community; everyone that shares their ideas, notebooks and datasets. I have enjoyed learning so much from you all.

Special thanks to:
@carlmcbrideellis, @ambrosm, @lucamassaron, @adamwurdits, @vpallares, @mfedeli, @fergusfindley, @teckmengwong, @vad13irt, @remekkinas 

Bring on the next TPS competition!
