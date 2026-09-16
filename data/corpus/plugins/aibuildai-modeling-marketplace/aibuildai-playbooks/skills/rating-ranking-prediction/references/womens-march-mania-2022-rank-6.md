# 6th Place Solution

Competition: womens-march-mania-2022
Rank: #6
Source: https://www.kaggle.com/c/womens-march-mania-2022/discussion/317105

For the third year in a row, being aggressive in predictions worked out **much** better in the women's competition than in the men's competition. This probably shouldn't come as a surprise, considering the relative lack of parity in women's basketball compared to men's over the past decade.

I used a model exclusively based in XGBoost, and used a good chunk of my time hyperparameter tuning to optimize the results of my CV. 

- The target feature in the train CV was margin of victory

- The accuracy measure was mean absolute error

- Historic win percentages based on final point spreads were used to map the margin of victory to a win probability

- Aggressive override #1: map any game with a win probability >= 92.5% to 99% likelihood. This *almost* burned me in the NC State/Notre Dame game, but Raina Perez was my hero! I didn't do the math, but I may have finished in the top 5 if I had set this to 99.999% instead. Honestly, I'd rather not go through this exercise because it'll probably cause me too much anguish if my loose hypothesis is true :)

- Aggressive override #2: All 1's and 2's were set to a 99.9% likelihood in the first round, and 3's were set to a 99% likelihood

- Aggressive override #3: All 1's were set to a 99% likelihood of winning against whatever 8/9 seed they played in the second round. At the last second, I decided to stay away from doing this with 2's against the 7/10 seed - relatively lucky given Iowa and Baylor both going down.

At the end of the day, finishing just outside the top 5 was primarily a result of UConn going to the championship game, as my model had them as underdogs against NC State and Stanford. Lesson learned: I need to implement a Geno Auriemma manual override next season!
