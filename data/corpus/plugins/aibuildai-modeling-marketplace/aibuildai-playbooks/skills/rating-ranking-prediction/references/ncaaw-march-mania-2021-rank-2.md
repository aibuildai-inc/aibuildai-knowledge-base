# 2nd Place Solution

Competition: ncaaw-march-mania-2021
Rank: #2
Source: https://www.kaggle.com/c/ncaaw-march-mania-2021/discussion/230705

Congratz to everybody ! I always enjoy taking part in those competitions, although they revolve a lot around randomness, this year was a lucky one for me and I can't complain !

My solution is available here : https://www.kaggle.com/theoviel/ncaaw-model

### Model

I use the same approach as in my public kernel.

The model is a simple logistic regression using 4 features : 
- Seed Difference
- FiveThirtyEight rating difference
- Win Ratio difference
- Average score gap difference 

The reasoning is that I wanted a model that has "safe" predictions in order to survive upsets. 
I knew however such model alone won't reach the gold zone, so to counter-balance this I override a lot of predictions.

### Overriding predictions

I use a really agressive strategy for my 2nd place submission, and a safer one for the other which scored #15. I analysed FiveThirtyEight brackets and simulated a lot of matches in order to pick the teams. I will only go in the details of the risky one, which is the following :

- Picked 11 teams that would win their first match. I didn't want to use the 16 teams seeded <= 4 as I had a feeling there would be an upset. Which happened, thanks Arkansas. 

- Stanford and Baylor beat every team seeded 3 or higher
- Connecticut and South Carolina  beat every team seeded 4 or higher
- Maryland wins beats every team seeded 7 or higher


I used p=0.99999 for overriding. So one mistake meant I was out. In total I overrode 22 matches. Most of them went smoothly except for three :
- Baylor vs Michigan went to OT.
- I messed up my code and ended up overwriting Stanford vs Arizona. I did not mean to do that, as it was basically a coin-flip.
- Texas A&M almost lost their first match.

I estimate the probability of this strategy to have less than 1 chance out of 10 to work. But I knew that me having my 22 guesses correct would mean a very nice finish :)

*Thanks for reading !*
