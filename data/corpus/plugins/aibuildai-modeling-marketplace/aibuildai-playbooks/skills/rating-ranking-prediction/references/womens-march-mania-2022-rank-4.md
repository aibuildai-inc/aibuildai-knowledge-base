# 4th Place Solution

Competition: womens-march-mania-2022
Rank: #4
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/317183

First, I’d like to thank the Kaggle team for hosting this exciting competition for another year. Congratulations to everyone, as we all gained experience this year that would make our models better next year. It’s the third year for me to participate in the NCAA competitions. 2020 was a big disappointment as the tournament got canceled at the last minute due to COVID. This year, I had the luck to end up in the top 1%. The journey to the top was full of excitement.

My model is a carry-forward over raddar’s well-known 2018 1st place solution. Shout out to raddar! Your code has inspired many over the years. My work mainly focused on the following aspects:

1. Feature Engineering
To give the model more opportunities to extract signals from historic data, my first step was to create more compound features using various evaluation metrics. Some past winners went against this method, saying that simple metrics worked the best, while I thought they may not have found those that truly worked. I found a full page of team evaluation metrics from NBAstuffer.com via the link below. After blending most of them in, I ended up having more than 100 features to choose from.
https://www.nbastuffer.com/analytics-101/team-evaluation-metrics/

2. Feature Selection
Feature selection is always a tedious but rewarding process for me. It was easy to get rid of those correlated features, but choosing from what was left wasn’t fun. I started by using SHAP to explain the feature importance of the XGBoost model to reduce the number of features to 50. Then through trial and error, I manually removed another 30 before seeing the CV loss at the bottom.

3. Games Overriding
One shouldn’t be surprised if he/she learned that the winners had a lot of games overridden. Overriding the 1st round games has become a must for everyone who wants to stay in the game. This year the #14 Jackson St. almost upset the #3 LSU. They led by 10 points in the 4th quarter but didn’t hold up. If that had happened, many would have disappeared on the LB. The other game was #2 Baylor vs #10 South Dakota. I only had it overridden in one of my submissions, as in the past 11 years, there was only one upset between #2 and #10 seeds. But this year alone two more had been added to the record, the other being #10 Creighton beat #2 Iowa. Next year you can’t be more cautious with #2 vs #10. I took more risks by betting over a few more games. For an entire week, I was anxious about #1 NC State vs #5 Notre Dame, as Notre Dame's 108 points in their last game were no kidding. Raina Perez's final steal saved NC State as well as sent me to #1 on the LB after the Sweet 16.

4. Final Rounds
Since the Elite Eight, my model had been on auto-pilot, without overrides. The battle for a position in the top 5 in the last few games was more fierce than I expected. I was jealous of others who had an insane 0 loss in the Elite Eight and Final Four, while my model gave me considerable losses over UConn alone. I dropped to #6 before the title. No complaints cause those bets between strong seeds were way too risky for me. I ended up going back to the 4th place as some all-in bets hurt two unlucky friends.

Again, congrats to everyone. For those who happened to miss a better medal or prize, don’t be disappointed. You are as good as the next level, just a bit short of luck this year. May luck with us all in 2023!

Please upvote if you found this post helpful. Thanks.

[code](https://github.com/wii365/NCAAW2022)
