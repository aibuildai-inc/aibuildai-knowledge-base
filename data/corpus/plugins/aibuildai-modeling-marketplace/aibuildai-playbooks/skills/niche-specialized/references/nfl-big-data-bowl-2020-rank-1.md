# 1st place solution The Zoo

Competition: nfl-big-data-bowl-2020
Rank: #1
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400

We are glad to publish the solution write-up of The Zoo by @dott1718 and @philippsinger.

We want to sincerely thank the hosts and Kaggle for making this competition possible. We had a lot of fun crafting our solution as it was necessary to think a bit out of the box and come up with something that really reflects the situation on the field. An extra thanks goes to Michael Lopez for actively participating in all the discussions and activities around the competition. That did add motivation to improve and believe that we can bring some value to NFL analytics. Can’t remember the last time we’ve seen such involvement of a host into the competition.

There was little problem with the data (2017 measurement differences were disclosed) and there was a nice correlation between CV and public LB. There was also no real chance to cheat as private LB will be on future data. We also want to thank all competitors for not exploiting the possible leak in public LB.

We really hope there won’t be any surprises on the private LB data and we hope our kernels will run through. In these types of kernel competitions there is always the risk of something failing, which would be devastating, of course. 

Regardless of what happens, we are really proud of our solution and strongly believe that it can be a valuable asset to future endeavors in NFL analytics.

**TL;DR:** It’s a 2d CNN based on relative location and speed features only.

###Solution motivation
Few words about how we came up with the model structure. To simplify we assume a rushing play consists of:
- A rusher, whose aim is to run forward as far as possible
- 11 defense players who are trying to stop the rusher
- 10 remaining offense players trying to prevent defenders from blocking or tackling the rusher

This description already implies connections between which players are important and which might be irrelevant, later we proved it to be the case on CV and LB. Here is an example of play visualization we used (based on the modified kernel from Rob Mulla [1])



If we focus on the rusher and remove other offense team players, it looks like a simple game where one player tries to run away and 11 others try to catch him. We assume that as soon as the rushing play starts, every defender regardless of the position, will focus on stopping the rusher asap and every defender has a chance to do it. The chances of a defender to tackle the rusher (as well as estimated location of the tackle) depend on their relative location, speed and direction of movements.




Another important rule we followed was not to order the players, because that would force an arbitrary criteria into the model, which will not be optimal. Besides, the picture from above gives us the reason to believe each defender should be treated in a similar manner.

That points to the idea of a convolution over individual defenders using relative locations and speeds, and then applying pooling on top.

At first we literally ignored the data about 10 offense players and built a model around the rusher and defenders, which was already enough to get close to 0.013 on public LB. Probably with proper tuning one can even go below 0.013.

To include the offense team player we followed the same logic - these 10 players will try to block or tackle any of the defender if there is a risk of getting the rusher stopped. So, to assess the position of a defender we want to go through all the offense team players, use their location and speed relative to the defender, and then aggregate. To do so, we apply convolution and pooling again. So good old convolution - activation - pooling is all we needed.

###Model structure
The logic from above brought us to the idea of reshaping the data of a play into a tensor of defense vs offense, using features as channels to apply 2d operations.

There are 5 vector features which were important (so 10 numeric features if you count projections on X and Y axis), we added a few more, but they have insignificant contribution. The vectors are relative locations and speeds, so to derive them we used only ‘X’, ‘Y’, ‘S’ and ‘Dir’ variables from data. Nothing else is really important, not even wind direction or birthday of a player ;-)

The simplified NN structure looks like this:



So the first block of convolutions learns to work with defense-offense pairs of players, using geometric features relative to rusher. The combination of multiple layers and activations before pooling was important to capture the trends properly. The second block of convolutions learns the necessary information per defense player before the aggregation. And the third block simply consists of dense layers and the usual things around them. 3 out of 5 input vectors do not depend on the offense player, hence they are constant across “off” dimension of the tensor.

For pooling we use a weighted sum between both average and max pooling with average pooling being more important (roughly 0.7). In earlier stages of the model, we had different kinds of activations (such as ELU) as they don’t threshold the negative weights which can be problematic for the pooling, but after tuning we could switch to ReLU which is faster and had similar performance. We directly optimize CRPS metric including softmax and cumsum.

For fitting, we use Adam optimizer with a one cycle scheduler over a total of 50 epochs for each fit with lower lr being 0.0005 and upper lr being 0.001 and 64 batch size. We tried tons of other optimizers, but plain Adam is what worked best for us. 

###CV
We were quite fortunate to discover a really robust CV setup. Probably, we will never have such a nice CV again. In the end, it is quite simple. We do 5-fold GroupKFold on GameId, but in validation folds we only consider data from 2018 (similar to how Patrick Yam did it [2]). We saw very strong correlations between that CV and public LB as 2019 data is way more similar to 2018 data compared to 2017 data. Having the 2017 data in training is still quite crucial though. As we are using bagging on our final sub, we also bagged each fold 4 times for our CV, meaning our final CV is a 5-fold with each fold having 4 bags with random seeds.

Having such a strong CV setup meant that we did not always need to check public LB and we were quite confident in boosts on CV. We actually had quite a long period of not submitting to public LB and our improvements were all gradual. Based on given correlation, we could always estimate the rough LB score. You can see a plot of some of our CV and LB models below. The x-axis depicts the CV score, and y-axis respective LB score. Blue dots are models actually submitted to LB, and red dots are estimates. You can see that we lost the correlation only a tiny bit in the end, and our theoretical public LB score would have been below 0.01200. Our final CV for 2018 is around 0.012150.



###Data processing
As we assume most people did, we adjusted the data to always be from left to right. Additionally, for training we clip the target to -30 and 50. For X,Y and Dir there is no other adjustment necessary, however, as most have noted, there are some issues with S and A. Apparently, the time frames were slightly different between different plays.

For S, the best adjustment we found is to simply replace it with Dis * 10. A is a bit more tricky as there is apparently some form of leak in 2017 data (check the correlation between rusher A and target). So what we did is to adjust A by multiplying it with (Dis / S) / 0.1. That means we scale it similarly to how we scale S. After all, A only has a tiny signal after this adjustment, and one can easily drop it. As we rely on relative features in the model, we don’t apply any other standardization.

###Augmentation &amp; TTA
What worked really well for us is to add augmentation and TTA for Y coordinates. We assume that in a mirrored world the runs would have had the same outcomes. For training, we apply 50% augmentation to flip the Y coordinates (and all respective relative features emerging from it). We do the same thing for TTA where we have a 50-50 blend of flipped and non-flipped inference.

###Code optimization
We decided quite early that it is best to do all the fitting within the kernel, specifically as we also have 2019 data available in the reruns. So we also decided early to spend time on optimizing our runtime, because we also knew that when fitting NNs it is important to bag multiple runs with different seeds as that usually improves accuracy significantly and it removes some of the luck factor.

As mentioned above, we use Pytorch for fitting. Kaggle kernels have 2 CPUs with 4 cores, where 2 of those cores are real cores and the other 2 are virtual cores for hyperthreading. While a single run is using all 4 cores, it is not optimal in terms of runtime, because you cannot multiprocess each operation in a fit. So what we did is to disable all multithreading and multiprocessing of Python (MKL, Pytorch, etc.) and did manual multiprocessing on a bag level. That means we can fit 4 models at the same time, gaining much more runtime compared to fitting a single model on all 4 cores.  

Our final subs fit a conservative number of 8 models each, having a total runtime of our subs at below 8500 seconds.

###What didn’t work
- Transformers and multihead attention, which seem to approximate the dependencies we explicitly use. We mainly focused on trying out attention to include offense-offense and defence-defence dependencies.
- LSTM instead of CNN.
- Adding dependencies like offense-offense and defence-defence explicitly.
- As soon as all the inputs are vectors, it seems tempting to try complex numbers based NNs. There is even a nice paper and a github repo available with the math and implementation of complex number version of all the layers we are using [3], but in keras. However, all the attempts we’ve made to limit CNNs to vector operations gave worse results.
- Going deeper and wider.
- CNN adjustments known from CV like Squeeze-and-Excitation layers or residual networks.
- Voronoi features.
- Weighing 2018 data higher than 2017 data.
- Multi-task learning, label smoothing, etc.
- Other optimizers, schedulers, lookahead, etc.

###Final subs
Our first sub is our best model fitted on an 8-fold with picking the best epochs based on CV using 2018 and 2019 data (in the rerun, only 2018 in public LB). This model currently has 0.01205 public LB. Our second sub is using full data for fitting with fixed epochs (no early stopping). It currently has public LB 0.01201.

In private reruns we incorporate 2019 data into training and we hope that all goes well, but you never know.

P.S. Don’t forget to give your upvotes to @philippsinger as well - this model is a great example of teamwork.


[1] https://www.kaggle.com/robikscube/nfl-big-data-bowl-plotting-player-position
[2] https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314
[3] https://arxiv.org/abs/1705.09792
