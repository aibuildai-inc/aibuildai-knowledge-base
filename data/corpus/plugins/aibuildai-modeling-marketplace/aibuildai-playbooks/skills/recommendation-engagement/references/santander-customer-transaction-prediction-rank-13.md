# #13 Place - Solution and Takeaways

Competition: santander-customer-transaction-prediction
Rank: #13
Source: https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89083#latest-517385

Almost all of our team's approach has already been discussed by other teams. Much of the credit goes to my teammates @wrosinski @christofhenkel @lukeeee and especially @returnofsputnik who gets the credit for finding the magic first.

High level we:
1. Utilized the unique counts feature using the train and non-sythetic test set for 0.91+
2. Realized that we could train single models on each individual feature (we didn't use naive bayes as I've seen others did.)
3. @christofhenkel trained a handful of NN models [that he posted about here](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89009#latest-513694)
4. Tuned LGBM and XGBoost parameters for individual single feature models We never could get catboost to perform well. We also trained a much of other model types (mostly sklearn) that we didn't end up using.  0.92+
5. Once we had all the models we knew we wanted to use, we trained each model with 4 different seeds and 5 different bagging rounds with 8-fold cross validation. 0.92++
6. Some strategic (and stressful) consideration was made on how to blend the NN and single feature models for our final few submissions. We ended up using a variation of blends with 70% NN and 30% single feature models. Thinking through this was really important since we only had a few LB submissions to work with.

More importantly I'd like to post some of my takeaways from the competition. [I did something similar after the NFL Punt competition](https://www.kaggle.com/c/NFL-Punt-Analytics-Competition/discussion/78239) and I think it was helpful. After so many grueling hours working on (and thinking) about this competition I'm sure many are left feeling frustrated - personally I've found it helpful to take a step back and write out some lessons learned. This was the first ML competition that I really took seriously so my insights are taken from the viewpoint of a newbie.


Things I learned and will take into the next competition:
- Learn from others, both on the discussion boards and kernels. Most of the magic was shared, but it wasn't spoon fed. You still had to dig into the data. Generally I found the discussion boards were pretty welcoming to newbies like myself.
- At the same time- don't spend too much time on the discussion boards and don't take anything on the discussion boards as gospel. Did someone say they tried something and it didn't work? Maybe they didn't do it correctly- so try it yourself. Worst case scenario it doesn't work and you've gotten better at coding, best case scenario you find something new.
- The discussion boards can be fun - [My personal favorite was everyone's shock and disbelief when the first team cracked 0.907! ](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/83121#latest-488751) We had to keep changing the title as the top score kept going up. Also @CPMP 's new 'uncle' title never fails to make me chuckle.
- Create a lot of experiments. Sometimes instead of asking if something would work- its best to just try it! Kaggle kernels are free so kick one off to test your hypothesis. If it fails make the kernel public so others can learn too. [I posted about unique counts almost a week before actually knowing it was part of the magic](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/87320#latest-506231). I just needed to try more experiments.
- Still feeling stuck? Post about it- I found my posts got more discussion flowing if I had already tested a hypothesis and showed my findings.
- Think about your total submission count- this is important if you intend to eventually team up. At the beginning I was trying to submit every day 3 times a day just to move to a higher 0.901 position. What I didn't realize is that the total team submissions must combine to less than 3 a day **combined**- so each submission is valuable.
- Team up with people smarter than you if you can. I know I got really lucky in this aspect and I learned a lot from my teammates. I think I would've found the magic on my own, and possibly even cracked 0.91- but there is no way I would've been able to get anywhere close to 13th place without my teammates.
- Try to make code clean and reusable. If you are going to team up the someone else needs to be able to understand what your code does. I need to get better at this.
- Know when to ditch experiments and move on to stacking/ensembling. The last weekend I was still trying to create some features to boost our score but at a certain point we made a decision as a team to quit experiments and to only submit things to the LB we thought would give us a higher score.
- Lastly - **Kaggle competitions are HARD.** If you really want to compete you are going to spend a lot of hours, get really frustrated, and probably not even do that well (I experienced that in the NFL competition). I have a whole new appreciation for kagglers that have been able to consistently finish at the top of competitions like this. They are very smart people for sure- but more than that they have incredible grit to stick with something that can be so frustrating and stressful.

Thanks to Kaggle and Santander and the kaggle community for the fun competition! I learned a lot.
