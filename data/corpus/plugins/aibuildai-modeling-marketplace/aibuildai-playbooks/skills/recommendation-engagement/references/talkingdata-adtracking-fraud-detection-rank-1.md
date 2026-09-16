# 1st place solution

Competition: talkingdata-adtracking-fraud-detection
Rank: #1
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475

We'd like to thank TalkingData and Kaggle for organizing this exciting competition. This competition gave us a fantastic opportunity to learn how to deal with very large table data. <br>
<br>
Here is our solution. <br>
<br>
**Strategy:** <br>
Our solution heavily depends on negative down-sampling [1, 2], which means we use all positive examples (i.e., is_attributed == 1) and down-sampled negative examples on model training. We down-sampled negative examples such that their size becomes equal to the number of positive ones. It discards about 99.8% of negative examples, but we didn't see much performance deterioration when we tested with our initial features. Moreover, we could get better performance when creating a submission by bagging five predictors trained on five sampled datasets created from different random seeds. This technique allowed us to use hundreds of features while keeping LGB training time less than 30 minutes. <br>
<br>
**Features:** <br>
First, we started from features from Kernels. On Thanks pranav84 and other Kagglers for sharing their awesome insights! We did feature engineerings using all the data examples instead of the down-sampled ones. <br>
- five raw categorical features (ip, os, app, channel, device) <br>
- time categorical features (day, hour) <br>
- some count features <br>
Then, we created a bunch of features in a brute-force way. For each combination of five raw categorical features (ip, os, app, channel, and device), we created the following click series-based feature sets (i.e., each feature set consists of 31 (=(2^5) - 1) features): <br>
- click count within next one/six hours <br>
- forward/backward click time delta <br>
- average attributed ratio of past clicks <br>
We didn't do feature selection. We just added all of them to our model. At that point, our LGB model's score was 0.9808. <br>
<br>
Next, we tried categorical feature embedding by using LDA/NMF/LSA. Here is the pseudo code to compute LDA topics of IPs related to app. (LDA is latent Dirichlet allocation)<br>
<pre>apps_of_ip = {}
for sample in data_samples:
  apps_of_ip.setdefault(sample['ip'], []).append(str(sample['app']))
ips = list(apps_of_ip.keys())
apps_as_sentence = [' '.join(apps_of_ip[ip]) for ip in ips]
apps_as_matrix = CountTokenizer().fit_transform(apps_as_sentence)
topics_of_ips = LDA(n_components=5).fit_transform(apps_as_matrix)
</pre>
We computed this feature for all the 20 (=5*(5-1)) combinations of 5 raw features and set the topic size to 5. This ended up with 100 new features. We also computed similar features using NMF and PCA, in total 300 new features. 0.9821 with a single LGB. <br>
<br>
After that, we removed all raw categorical features except app since we supposed embedding features cover information available from them. Surprisingly, this minor change made our public LB score jump up from 0.9821 to 0.9828. Actually, we don't know what causes this significant score improvement. <br>
Besides features mentioned here, we created higher dimensional LDA features and features that try to address the duplicate sample problem. These features somewhat improve our public LB score. <br>
<br>
**Models:** <br>
We used day 7 &amp; 8 for training and day 9 for validation, and chose the best number of iterations of LGB. Then, we trained a model on day 7 &amp; 8 &amp; 9 with the obtained number of iterations for creating submission. After we finished feature engineering, flowlight's five-bagged LGB model reached 0.98333 on public LB (and 0.98420 on private LB), which was trained on 646 features. As far as we remember, the memory usage for training this model was less than 100GB (&lt;64GB will be possible with minor code modification). <br>
I implemented a simple three layer NN model as some kernels do. It scored worse than LGB models by 0.0013 points with 0.005 down-sampling rate at first. Then, I realized it should be trained with more negative data samples, probably. However, we didn't afford to use so many examples because of the massive amount of our features (and my bad implementation, disk space, no GPU, close deadline, etc...). The final three-bagged NN model scored 0.98258 on public LB. <br>
We made our final submission with a rank-based weighted averaging. It is composed of seven bagged LGB models and a single bagged NN  It scored 0.98343 on public LB. <br>
<br>
<br>
[1] <a href="http://quinonero.net/Publications/predicting-clicks-facebook.pdf">Practical Lessons from Predicting Clicks on Ads at
Facebook</a> <br>
[2] <a href="https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf">Ad Click Prediction: a View from the Trenches</a> <br>
<br>
EDIT: Added some extra explanations based on frequently asked questions in comments.
