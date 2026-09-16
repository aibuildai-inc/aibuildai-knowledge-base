# TOP1 : Additional Information and Unique Techniques

Competition: jane-street-market-prediction
Rank: #1
Source: https://www.kaggle.com/c/jane-street-market-prediction/discussion/229623

国内的朋友可以[看这里](https://zhuanlan.zhihu.com/p/355606168)🙈
[kaggle Kernel](https://www.kaggle.com/xiaowangiiiii/current-1th-jane-street-ae-mlp-xgb)
[ github](https://github.com/MingjieWang0606/-Jane-Street-AE-MLP-xgb-TOP1)

When we finally submitted it, we made some changes to yirun's code. But most of them have not changed, you can check our notebook and [the yirun discussion](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348) for details.

The official data provided 5 sets of income indicators, namely resp_1, resp_2, resp_3, resp_4 and resp. Our final benefit is calculated according to the value of resp. please check [here.](https://www.kaggle.com/christoffer/rough-estimate-of-resp-i-timeframes)

According to resp_1, the income obtained by resp_2 exchange is more stable, and the t value is high
According to resp_3, the exchange may have a higher t value, and the p value tends to be higher
According to resp, resp_4 transactions may achieve high p-values, but t-values ​​are often very low
From the above conclusions, we found that by using resp_3, we can obtain a model that is both stable and highly profitable. If we directly use resp_3 when predicting and training, we are likely to get higher benefits than using resp to predict!
Regardless of the model, submit resp_3 to increase the point! This first conclusion is valid most of the time whether it is online or offline.


This is the income curve obtained when training the neural network. The abscissa is epoch//5, and the ordinate is the utility score. For the gains obtained by different resp, it can be seen that the gains obtained by resp_1 and resp_2 are always low, while resp_3 , resp_4, resp gains even higher gains. The curves listed here are those obtained when the model converges relatively stably. In fact, by selecting different verification sets, different modes can be observed. When the data distribution is completely different, resp_2 can achieve better returns!

Mean or mode?
In fact, through offline analysis and observation, it can be found that in the pre-test, if multiple targets are used during training, the gains obtained by taking the mean and mode are actually the same, but for some special cases, the mean or mode is often surprisingly effective. In view of the serious uncertainty here, we have not done much research here.

Optimal transaction threshold
Offline experiments have concluded that a threshold between 0.51 and 0.52 can bring higher benefits. One of the final submitted models uses 0.51 as the threshold. I also hope that this discovery can be used in the next six months. Come on positive income.

How to improve the model effect?
Using more data (weight=0), here is where we are lacking, but I believe that other teams have not done too well. It is not ruled out that some teams have done some processing and achieved results, but they have done some processing and It does not mean that they have used the data. It takes a lot of experiments to verify that it is correct before drawing conclusions.
Reduce the number of transactions. In view of the penalty mechanism of the evaluation standard, giving up risky trading opportunities may result in an increase in scores. But here is also more metaphysical, and no convincing method that can improve scores has been found.
Feature engineering. More feature engineering can make it easier for the model to learn the patterns in the data, but the subsequent severe overfitting.
Final model
The data of the game are anonymous features (130 columns). Feature engineering is not completely useless, but it is often thankless. In order to maintain the simplicity of the solution, the final submitted model did not include feature engineering.
In fact, there are some feature projects that can improve scores offline, but adding online models will lower the scores. We hope to keep only the strong conclusion that both online and offline are improved.
Readers who have read here may find that we have not used any particularly complicated methods. In fact, most of my energy is spent on observing the performance of data and models. Complex models or methods have also been tried, but the results are often not good, online and offline are inconsistent, or not significant. I believe everyone will have an intuition that the final winning model may not be that complicated. This is also the principle we followed in the final submission.

Our final model is AE+MLP + XGBOOST (100 round), and each model is trained on three seeds. The scores obtained in the public list are 99xx and 97xx respectively, and the scores in the private list are 54xx and 52xx.

I'm lazy, so I use Google Translate directly. Please let me know if there is something unclear about the expression. :)
Thanks to my teammates!!!Without them, I can't do anything.
