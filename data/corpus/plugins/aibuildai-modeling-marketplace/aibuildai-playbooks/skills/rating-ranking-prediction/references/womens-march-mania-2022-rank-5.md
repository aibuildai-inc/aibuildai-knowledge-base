# 5th Place Approach

Competition: womens-march-mania-2022
Rank: #5
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/317961

Thanks to Kaggle for hosting this competition again. This competition is part of what makes March my favorite time of year and I appreciate all of the work it takes on the back end to set up. 

Here's a bunch of info on my submission with a link to the code and a tldr at the end, please let me know if you have any questions.



# Background on you/your team

My name is Taylor Merry and I am a data scientist living in Seattle, WA.

**What is your academic/professional background?**
BS in Statistics from University of Washington. 2 years experience as data scientist in industry.

**Did you have any prior experience that helped you succeed in this competition?**

Yes, I have participated in many of the previous iterations of the Kaggle March Madness competitions. I am also a big college basketball fan and follower of sports analytics.

**What made you decide to enter this competition?**

I enjoyed the past iterations of this competition

**How much time did you spend on the competition?**

This year, I had a lot of the code already set up from previous entries, so only around 20 hours or so

# Summary

**Training method(s) you used**

Normal distribution (parametric model) from publicly available rating systems


**The tool(s) you used**
I used publicly available rating systems and a normal distribution to convert from points differential to win probability

**How long it takes to train your model**

No training, notebook runs quite quickly

# Features Selection / Engineering

**What were the most important features?**

The publicly available rating systems

**How did you select features?**

Looked through a few rating systems and picked the ones that looked the best

**Did you make any important feature transformations?**

Not really unless you would count making manual adjustments to some of the teams’ ratings

**Did you use external data? (if permitted)**

Yes, see references

# Training Method(s)

**What training methods did you use?**

None, just used a parametric model using publicly available rating systems and research to set the parameters

**Did you ensemble the models?**

I used two different rating systems and did a weighted average

**If you did ensemble, how did you weight the different models?**

I decided to weight the Moore power ratings as 60% and the Talisman Red ratings as 40% because I had used the Moore ratings before and knew they were reliable

# Interesting findings

**What was the most important trick you used?**

Knowing that UConn played most of the season injured so they’d be highly underrated

**What do you think set you apart from others in the competition?**

Boosting UConn, not giving heavy favorites 100% win probability (Baylor, Iowa)

**Simple Features and Methods**

My model was already quite simple. I think for this competition, it’s best to have a simple prediction model and then use some intuition on how you should adjust your probabilities.

# Model Execution Time

My predictions used no training, the notebook only takes a few seconds to run.

# References

Ratings I used:
- [Sonny Moore](https://sonnymoorepowerratings.com/w-basket.htm)
- [Talisman Red](https://talismanred.com/ratings/whoops/)

Sources for Kenpom using 11 points as standard deviation for men’s college basketball ([source 1](https://www.reddit.com/r/CollegeBasketball/comments/5xir8t/calculating_win_probability_and_margin_of_victory/), [source 2](https://www.google.com/url?q=http://practicallypredictable.com/2018/03/13/ken-pomeroy-ratings-and-first-round-upset-picks/&sa=D&source=docs&ust=1649541627318853&usg=AOvVaw1DMKq_kvBc8Ax-paTOXuSV))




#Tldr:

I would say the biggest factors leading to my success in this competition were (in order):
1. Not giving heavy favorites 100% win probability which helped me survive the Baylor and Iowa losses
2. Luck (my men’s entry finished in the 500s and I guess the winner of the men’s competition accidentally used 2018 data???)
3. Knowing UConn was injured and thus would be underrated (I don’t think I would have placed if UConn didn’t make the title game)


[Link to code](https://www.kaggle.com/code/tmerry/2022-women-s-march-mania-predictions)
