# 6th Place Solution

Competition: march-machine-learning-mania-2023
Rank: #6
Source: https://www.kaggle.com/c/march-machine-learning-mania-2023/discussion/400709



**Background** 

I graduated from Heriot-Watt University in Edinburgh, Scotland in 2006 with a degree in Actuarial Mathematics and Statistics. 

Since then, I have been working for multiple banks in the United Kingdom and Australia predominately in credit risk modelling. This has involved creating mathematical models to help the organisation make the correct decisions. In addition I have also looked at regulatory models and forecasting of provisions. I am currently a Director, Analytic Consulting in an analytics company based in Madrid.

This competition has become an annual hobby for me. I first discovered it in 2017 which is when I found out what March Madness was. In that competition I came 5th but I wasn’t happy with the approach and have changed it since then.

This year I spent a bit more time than usual because I moved my processes from running on my local machine to running on Kaggle. As part of that, I have reviewed and updated all my old codes to make sure they were doing what they should have and updated the process. This includes updating to use R code rather than using the SQL library which is how I produced some of some steps previously. I have included my code here (https://www.kaggle.com/code/djscott1909/march-madness-2023-ds) which you can review but please be kind. I have taught myself R during these competitions so I apologise for any inefficient coding. 

**Summary**

My code breaks down into multiple sections. The first 6 sections of my code are included to use the data provided by the competition and looking at different ways to generate variables. This code includes making a range of my own ratings for teams. They include:

- **A Simple rating system** – This creates a rating for all teams based on their season results and using the margin of victory in each game to generate a rating. 
- **An ELO rating** – The same concept except it uses the ELO process (https://en.wikipedia.org/wiki/Elo_rating_system) to generate the teams rating.
- **Summary statistics** – Using the information to generate some statistics which could be beneficial from the games. This includes Win percentage, Points for and against, Strength of Schedule, quadrant 1 wins, etc.

The last two sections are focused on generating a unique model for the Men’s and Women’s competition. I use all the information available and don’t introduce any of the statistics which are only available in later years, i.e. I don’t use the detailed results dataset, location information (only if a team was home, away or neutral), 538, odds or use the 0/1 method to optimise. I have used both odds and 0/1 in previous years but with the changes this year I didn’t think it was worth it. I also haven’t used the override approach, i.e., setting 1 Seed to 100% chance of winning. In some years I am sure this will be beneficial but I want to simply see how the model performs. 

The model itself is using a stepwise regression. I start the model with no variables and the step process looks to add in the most predictive variable and repeats the process. I prefer to use a step process which also reviews the variables in the model and removes them if they no longer add value to the model. This helps to make sure all variables considered are truly adding value. 

My final models looked as follows:

**Men**


**Women**


**Interesting findings**

I decided to test some variable transformations in the model development. This included creating a Weight of Evidence binning (WoE) for the variables, removing outliers and normalising the variable. 

During the Warmup competition I was careful to consider only the development and leverage the Out of time to help inform which direction I should go next. During this time, it seemed to show that Normalising didn’t add much to the Women’s competition. I can only assume the distribution within some of the variables made this better but I didn’t expect to see a difference between the two. My plan for next year is to investigate this further. 

**Simple Features and Methods**

This year, with the changes make, I decided to create 2 models and not leverage the 0/1 approach with the 2 submissions. I thought that the reward wasn’t as high this year so I went a different way. 

I had a much simpler model which didn’t contain the WoE fields. Unfortunately, it appears to be too simple (or maybe was just unlucky this year). This model would only have been good enough for 329th so I might need to investigate alternatives for my 2nd model for next year. 

Since my approach to the model development is using a stepwise regression and developing a single model the model ends up fairly simplistic in comparison to others (I think). While the models may look like a lot of variables it is really the same variable with different WoE binning so the underlying characteristics are fairly small.

**Additional Note**

I always enjoy these competitions but I wanted to give a mention to the work done by @zdbradshaw (https://www.kaggle.com/competitions/march-machine-learning-mania-2023/discussion/397987). The work he did with the submissions which were made public made the final games much more enjoyable for me as I knew which results to cheer for. I really encourage people to share their solutions next year so a similar exercise can be done. It adds a lot to the experience for me. 

This year was a roll coaster of emotions (being 800+ after the 1st 4 games) but I got luckier as the competition went on and that is why I keep coming back to it. I hope it continues for many years to come.
