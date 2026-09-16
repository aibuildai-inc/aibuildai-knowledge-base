# Official #3 Solution

Competition: mens-march-mania-2022
Rank: #3
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/318302

NCAAM Tournament 2022
A.	MODEL SUMMARY
A1 Introduction
Competition Name: 	penwright
Leaderboard Score:	0.55903
Name: 			Patrick Enwright
Location: 		Victoria, BC, Canada

A2 Background
•	Bachelor of Management, University of Lethbridge; MBA, University of Calgary
•	Previously worked in Canada’s oil and gas sector; transitioned to aviation sector with a focus on commercial – network planning, pricing, scheduling. 
•	Currently an executive for a publicly traded Bitcoin mining company, and continue to consult for the aviation sector. 
•	I have entered the Kaggle MLMM competition dating back seven years. I really enjoy the format and scoring methodology. 
•	I spent 4-5 hours in researching prior year’s solutions, compiling the data, preparing the model parameters, and analyzing the appropriate metrics while using the data for 2018, 2019 and 2021. 
•	In particular, I reviewed last year’s “Team Beat Navy”, which included Dusty Turner and Jim Pleuss. Dusty and Jim provided a solution recap, and I included the specific data metrics that Team Beat Navy used in 2021.  

A3. Summary
I built my model using Excel, and applying a regression analysis of pre-tournament metrics against the historical results using the ESPN points scheme. for the previous three (3) tournaments (2021, 2019, 2018). After compiling the metrics and standardizing, I then determined the weighting of each metric through the coefficients from the regression and arrived at my power rating for each team. I then ran the power ratings for 2018, 2019 and 2021 through an elo model to determine the progression of the tournament (see tabs “2018_elo”, “2019_elo”, “2021_elo”). 
Same time, I did not use the end elo rating for the purposes of the power rating and subsequent probabilities that I submitted for the Kaggle competition; it was more to see what the results looked like.  For the Kaggle submission, I used the power rating as provided in the “calcs” tab, columns FG:FJ. 
Key factors in my model
•	Offensive efficiency, defensive efficiency, percentage of points from 3s, possessions per game, assist to turnover ratio
•	Winning percentage
•	Free throw rate
•	Average ranking across all Massey Ordinals
•	COL and AP Ranking from Massey Ordinals
•	Quad 1 wins and Quad 4 losses
•	Tournament Seed of Team
•	Schedule Ranking
The following average statistics from the last 3 games of the season
•	Offensive efficiency
•	Defensive efficiency
•	Possessions per game
External Data
I used data from the link below for team efficiency, percentage of points from 3s, possessions per game, and assist to turnover ratio, specific to the last 3 games prior to the March Madness tournament. 
•	https://www.teamrankings.com/ncb/rankings/

A4. Features Selection / Engineering
The most important features were the ones already identified by last year’s team, “Team Beat Navy”. That said, from the regression, the features that had the highest weighting are:
•	Defensive efficiency (last three games)
•	Offensive efficiency (last three games)
•	Quad 1 Wins 
•	Winning percentage
•	3-Pt as a percentage of total points scored
The weighting of the metrics is as follows:
YEAR	DESCRIPTION	METRIC	   WEIGHT
2022	LAST 3 GAMES	Off. Eff.	          38%
2022	LAST 3 GAMES	Def. Eff.   	-49%
2022	LAST 3 GAMES	Poss/Game	  10%
2022	CY	                        Winning %	  23%
2022	CY	                        3-Pt as a %	  21%
2022	CY	                        Ass2TO	          -1%
2022	CY	                        SoS	                  11%
2022		                        Massey Avg.	 16%
2022		                        AP	                  6%
2022		                        COL	                  6%
2022		                        Quad 1 Win	24%
2022		                        Quad 4 Loss	-5%


A5. Training Methods
My training method was evaluation of the most recent tournaments – 2018, 2019 and 2021. In previous competitions, I have evaluated my historical power ratings back 7-10 previous tournaments. However, I believe a larger training set of historical tournaments dilutes the key parameters that determine the ability to choose a winner versus a loser.

A6. Interesting Findings
My overall thinking of NCAA men’s basketball is that the game has transformed with the success of Steph Curry/Klay Thompson/James Harden and the emphasis on three-point shooting at the professional level. Given the self interest of NCAA players wanting to become professional, the college game has evolved to mirror the “optimal efficiency” of the offense and the percentage of points scored being three-pointers. 
With that thinking in the back of my mind, it was interesting to see the historical results (2018, 2019 and 2021) have such an emphasis on the offensive efficiency and 3-Pt as a percentage of total points scored. 
Another key element that my model methodology did differently was using the Massey Ordinals as a counter reference. In this regard, a team like Gonzaga with its top ranking was rated as an inverse. I believe this is important for creating an offsetting weighting, albeit a modest weighting, for lower ranked teams to potentially “pull off the upset”.
Quad 1 Wins was also interesting to consider. From my interpretation and calculation of Quad 1 Wins/Quad 4 Losses, the success of a team that not only performs well against other top teams, but also refuses to lose to Quad 4-calibre teams reaffirmed my belief that mentally tough teams do not lose to lesser teams. 

A7. Simple Features and Methods
Here are the results of my model that I submitted via the Mark McClure Kaggle Brackets for March Madness. 
 
Source: https://wncviz.com/demos/NCAA_Brackets/kaggle_brackets.html 

A8. Model Execution Time
I did not evaluate the time to train my model or adjust parameters. 

A9. References
Team Rankings data: https://www.teamrankings.com/ncb/rankings/
Team Beat Navy 2nd place solution: https://www.kaggle.com/competitions/ncaam-march-mania-2021/discussion/231665
