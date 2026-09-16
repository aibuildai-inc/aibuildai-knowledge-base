# #11 solution: what worked and what didn't

Competition: playground-series-s3e16
Rank: #11
Source: https://www.kaggle.com/c/playground-series-s3e16/discussion/416819

Congratulations to the winners and all the participants! Also, I would like to say thank you to Kaggle who hosts playground competitions, making it easier for the beginners to start working on real competitions in simplified form. This playground was very interesting because of how many things I tried and how many didn't work out. 

**My plan was as follows:**

- build a good baseline model using AutoML tools
- try different feature engineering techniques to capture more relationship between weight-related features
- try to generate more data and use it for training
- try to find some "tricks" like in previous competitions, like manually correcting the rows with noise (something we have discussed with @epiktistes a day earlier)

**I have tried the following and most of those didn't work out:**

- feature engineering with creating various ratio rows like "shell-to-total", "body condition index" didn't improve the score and took longer to train
- I generated 400000 additional samples and used it for training, which reduced score a little
- then I noticed that the predictions are always in the range from 3 to 19 month, so I tried to find the outliers manually and fix it. Unfortunately, that turned out to be impossible due to how similar crabs are after they reach certain age
- after that I tried to generate more samples for old crabs, but that one didn't work too

So I just thought that well, maybe this competition doesn't have "tricks" like these and it will be all about tuning the model. I was quite right and, to my surprise, the highest scoring submission was the first baseline I built with three LGBM and two XGBoost models, which I have tuned using FLAML. Unfortunately, I didn't select that one. Secondly, I messed up submission names and didn't write the CV score in description for some files, so I ended up selecting not the best model as a final submission. And finally, I had some more hypothesis I could test, like generating more samples for younger crabs, I didn't have enough time for those. However, the 11th place is good anyway.

**What have I learned from this one?**

1. Sometimes ideas seem to be great, but they don't work in practice. In this case, just stick to a model with the best score and tune it further.
2. As always trust your own CV, though this time the shake-up wasn't too bad.
3. Keep the work clean and organized using proper file descriptions with detailed explanations what have been done and what score has been reached. 
4. Spend some time on discussions with the community, a lot of ideas are discussed in public and implementing those often helps.

Thanks everyone who participated and see you in the next one. Big thanks to @epiktistes, @phongnguyen1, @oscarm524, @arunklenin, @ravi20076 and @pandeyg0811 for staying active and sharing a lot of valuable insights during this competition. Keep Kaggling!
