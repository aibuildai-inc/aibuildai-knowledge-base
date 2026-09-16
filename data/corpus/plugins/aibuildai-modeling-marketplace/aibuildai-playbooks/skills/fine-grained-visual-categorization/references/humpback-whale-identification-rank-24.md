# 24th place solution

Competition: humpback-whale-identification
Rank: #24
Source: https://www.kaggle.com/c/humpback-whale-identification/discussion/82359#latest-481822

My solution was mostly based on this [version][1] of @martinpiotte amazing work in the previous whale competition. I noticed many comments mentioning the variability of results obtained from Siamese networks, which I thought might be advantageous. 

At first I made no modifications to Martin's network, training it from scratch for 500 epochs. At that point it would consistently get between 0.895-0.905 on lb. Training for another 5-10 epochs resulted in a similar score but the distribution of predicted whales was much more variable though the scores remained very close. I figured this was due to randomness in how the augmentations were applied so I trained several dozen versions of this model with minor variations--changing image size from 224 up to 600 both grayscale and rgb. I would have added TTA, but I do not know how to do it in keras. After this I had about 40 sets of predictions which I used for a hard-voting scheme at each whale position. A simple average of the prediction scores from each model could get a 0.935 with the right threshold and adding this to the voting resulted in 0.941 on public lb. 

At this point the scores were not increasing and it seemed I had reached the limit of what this network was capable of distinguishing so I added classification and prototype results to the voting. I find the prototype approach really interesting so I wish I had more time to work on it, I was able to get a single prototype model up to only 0.872. Adding several of these other models to the vote got the final score.

Great work by everyone, I learned so much and I can't wait to see how you all approached this.


  [1]: https://www.kaggle.com/seesee/siamese-pretrained-0-822
