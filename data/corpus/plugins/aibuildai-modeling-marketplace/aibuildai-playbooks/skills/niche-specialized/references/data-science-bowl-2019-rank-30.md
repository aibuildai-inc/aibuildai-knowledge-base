# 30th Place Write Up

Competition: data-science-bowl-2019
Rank: #30
Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/128417

I'd like to share what I learned a lot from this competition with you.

# Our Approach(public 622th → private 30th)
- Trust CV and LB.
- LGB with about 500 features.
- Some features were dropped by adversarial score(0.65 or so).
- Train by regression and then optimize with nelder-mead.

# Late Submission
I realized adversarial validation was useless, and "Trust CV" is the best approach here after some late submissions.
Also, if I used about 2k features, maybe I could get prize as well as gold medal and make my team mates Grandmasters.
.png?generation=1580450134030141&amp;alt=media)
.png?generation=1580450231030144&amp;alt=media)

# How to get gold medal
- Generate about 30k features.
- Use same condition as evaluation for validation.
- Trust only CV(after confirmed if we can trust LB).

- Use a lot of features till CV saturated.
- Don't worry about Adversarial Validation after all.
