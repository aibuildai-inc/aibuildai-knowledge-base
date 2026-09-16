# 5th place story (pocket)

Competition: talkingdata-adtracking-fraud-detection
Rank: #5
Source: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56319

So it was on a rainy day, one month till the end of the competition, that me and mamas met.  
I was 6th publicLB, and mamas was 2nd. Mamas offered me a team merge, and I agreed.  
We had a nice (cheap) dinner at a restaurant in Tokyo.  
We started talking about our model, and yes, we couldn't stop talking.   
We talked so long that I nearly missed my train. (Who said this competition was named "talking"?)  
  
Then after the day after we teamed up, Michael and Danijel joined. Two of the most talented kagglers.  
We thought that this team-up would alleviate the weak point our team had, namely NN and experience.  
As you may know, Michael has a legendary win with his NN on the Porto Seguro competition, and  
Danijel has a lot of experience with only one gold medal left for his GrandMaster title.  
Happy with the team composition, we aimed high. Higher than I could have ever imagined.  
The top prize.  
  
Here is what I tried.  
  
When we met, mamas told me that he had a few hundred features(!), and was training on a 1.4TB RAM GCP machine (WTF!)  
I instantly understood that this competition was not a competition for my single model to shine.  
So I decided to dig really hard on exotic techniques.  
The two most important ones I tried (which failed) were  
A)Pseudo Labeling B)Data Augmentation  
  
A)Pseudo Labeling  
Did anyone succeed with this? I tried PL with lightgbm in many ways.  
Different types of objectives(binary, regression, xentropy), different amount of data to mix, soft labels, hard labels.  
Nothing exceeded the score I could achieve with a single run.   
Regression exceeded the first round score, but still lower than the single run with binary objective.  
  
B)Data Augmentation  
I tried some form of pseudo data augmentation used in the Recruit competition.  
https://www.kaggle.com/pureheart/1st-place-lgb-model-public-0-470-private-0-502  
For example, I divided the data into 4hour windows, and slid the window by 2 hours to make 2x data.  
ex) 0-4am, 2-6am, 4-8am... and so on, and I extracted the features from each of the window.  
This did not work either.  
  
So there I was, only with two weeks left till the end of the competition.  
I realized that I could not make a difference with modeling techniques, and started to focus on my single model. Since by then, mamas had a 0.9826 model. I knew I should make a simple, but diverse model that adds to the blend.  
What did that look like?  
  
C) Single model(0.9828private)  
Since mamas had a gigantic model, I focused on simplicity.  
I had roughly 30 features(including the original ones).  
The most important ones (if you have not noticed) are, 1)nextClicks 2)ip-nunique features  
I extracted every feature day-by-day. This was unlike mamas or Danijel's model, so I think it added diversity.  
The model was seed-averaged with 5 seeds for the final submission. (+0.0001)   
Trained on a GCP machine. 64CoreCPU, 416GB RAM.  
I could have used a smaller machine by coding more smartly, but for me, time(and score) was important than money.  
  
In the end, we threw in our models into Michael's blending NN.  
0.9836(mamas), 0.9828(pocket), 0.9827(Danijel), and a few NN from Michael and Danijel.  
Out comes 0.9840, 5th.  
  
We couldn't get the top prize, but at the same time, I am happy with my gold medal.  
Happy Kaggling, and thank you for reading.  

P.S. You will be hearing from other Superstars from our team, in a few days...
