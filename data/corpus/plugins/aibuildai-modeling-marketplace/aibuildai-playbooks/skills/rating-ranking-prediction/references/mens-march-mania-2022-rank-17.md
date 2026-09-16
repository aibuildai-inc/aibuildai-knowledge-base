# 17th place solution

Competition: mens-march-mania-2022
Rank: #17
Source: https://www.kaggle.com/c/mens-march-mania-2022/discussion/317903

I want to share my solution as I believe it uses a somewhat unconventional strategy. This year is the first time I participate in this competition and I have to say, it was really exciting to see the games play out. Sadly for me, I didn't make Gold or top 5, mainly because of the St. Peters upset (predicted 92%), and of the comeback of Kansas in the final (predicted only 55% for Kansas). I did not look at any discussions in the submission phase, as I wanted to make a unique model and not get biased by other ideas. I came up with the idea of using different Elo systems, and feeding the Elo values into a small fully connected neural network. The Elo systems were calculated with the k values [10, 15, 20, 30, 50] (for details on the k-values: https://en.wikipedia.org/wiki/Elo_rating_system), like this performance on different timescales can be measured. To account for close match results, which a plain Elo system won't capture, I also calculated Elo values using the final points divided by the total points as a game outcome. Lastly, I passed a value in indicating whether it is a tournament game or not, passed the day of the season in and concatenated the input with the output of an **LSTM**, that takes in the last 10 Elo changes from the last 30 days. And that's basically it. My other submission also used the logarithmic mean of the official rankings, but this solution had a worse score of 0.59573 (compared to my 17th with 0.58903). I used the same as my 17th place solution in the women's competition and placed 95th there. My model generally profited from upsets compared to the median expert submission.
To conclude:

**What I have used:**
- Elo systems
- LSTM
- Small neural net

**What can be improved:**
I didn't think of calculating in what round a match up happens. I believe using this would have favored underdogs even more in later rounds.

This is the notebook:
https://www.kaggle.com/code/fritzcremer/17th-place-solution/notebook

The code isn't cleaned up well and not commented, as I have written it in two sessions and was a bit short on time, but I might add some comments here and there in later versions. Lastly I want to thank the host for this fun competition!
