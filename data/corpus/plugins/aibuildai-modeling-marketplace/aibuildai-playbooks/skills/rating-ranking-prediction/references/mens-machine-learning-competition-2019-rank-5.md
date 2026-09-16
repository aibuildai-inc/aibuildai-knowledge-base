# 5th place solution

Competition: mens-machine-learning-competition-2019
Rank: #5
Source: https://www.kaggle.com/c/mens-machine-learning-competition-2019/discussion/89942#latest-525738

Model Summary


Part 1 - Competition Background
Competition Name: Google Cloud &amp; NCAA® ML Competition 2019-Men's
Team Name: Sharp Sports Analytics
Private Leaderboard Score: 0.43148
Private Leaderboard Place: 5th

Name: Jordan Sundheim
Location: Reno, NV
Email: jodsun5@gmail.com


Part 2 - Personnel Background
After four seasons of coaching college I decided I was more interested in sports statistics than in coaching sports. I went back to school and got a Master's in Computational Economics at Duke. This was my 5th time entering the contest and reading the forums and interacting with the other Kagglers was huge in my success. The series of articles written after the 2014 competition were quite informative and helped me improve my model. I decided to enter the competition this season because I got the itch. In fact, I was going to skip the competition this year but the day before the tournament I decided to enter. I spent as much time as I could,  but given that I decided to enter the day before the tournament this was the least amount of time I've spent on this competition in my five entries.

Part 3 - Model
Simple is better, especially when it comes to sports data. While particularly neural nets and other machine learning methods tend to be quite popular these days I stuck to the basics and used a generalized linear model (probit). By far the most important features in my model were offensive and defensive scoring efficiency. While many others in the competition relied on Ken Poms numbers I generated my own. To generate my predictions I used a postgreSQL database to store and manipulate the data and I used R to run the model. The actual execution time for the model was quite quick. The script I ran to make my submission took a matter of seconds.

Part 4 - Feature Selection
Features Ranked by Importance
Offensive Efficiency
Defensive Efficiency
Win percentage against the spread
Top 10 / Top 25 pre-season ranking
School in major conference - Binary 0/1

Feature selection was a huge part of my model. I used three different approaches to construct my model. 1) Keeping it basic - O/D Efficiency has been documented as a major driver, so I built one model based simply on those. 2) Box score stats - I just put all the box score stats into a model and saw how they did. 3) Thinking like a gambler - I added features such as win percentage against the spread, win streak (straight up), and pre-season rankings. My final model was a weighting of these three approaches. I manipulated all statistics in order to account for opponent strength.

Part 5 - Training Methods
I used leave 1 season out validation. I would build the models of 15 seasons of data, leave one season out, and then did that a total of 16 times to see how the model would perform out of sample. Model weights were quite ad hoc. I remember reading somewhere that 80/20 spread/data was good and I went with it.

Part 6 - Interesting Findings
Luck matters. I manually manipulated some probabilities and my success can be attributed to Iowa's first round victory. If they lost I would have been screwed. As far as generating quality numbers, the stats themselves aren't that useful. There is so much variation in the quality of opponents in college basketball that you need to account for who someone plays in order to appropriately use the statistics.

Part 7 - Simple Features and Methods
To begin with I think my model was quite simple. But to simplify it further, offensive and defensive efficiency are the drivers. Those two stats alone get you 90% of the way.

Part 8 - Model execution time
Generating the historical statistics does take some time, running the adjusted box score stats for each season back to 2003 took about 20 minutes. But the actual execution time for the models and the submission form were quite quick. Maybe 30 seconds to run everything once the statistics were in the correct format.

Part 9 - References
I heavily relied on the data provided by Kaggle. There were three pieces of data I gathered from outside sources.
Per-Season Top 25 Rankings - Sports Reference
https://www.sports-reference.com/cbb/seasons/2019-polls.html
Team win percentage against the spread - Vegas Insider
http://www.vegasinsider.com/college-basketball/
Point Spread Data - Sportsbook Review
https://www.sportsbookreviewsonline.com/scoresoddsarchives/ncaabasketball/ncaabasketballoddsarchives.htm

My solution files are attached in .zip format as per the directions
