# Private 21st - 1st Competition

Competition: nfl-big-data-bowl-2020
Rank: #21
Source: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119552

After reading and following Kaggle competitions for a few years I finally decided to take the plunge.  I am a huge New England Patriots fan and also love Deep Learning so this was the perfect competition for me.  I had a ton of fun and learned so much.  I am extremely glad I did it.  To anyone reading this who hasn't done a competition yet - just go for it!  The community is amazing and I promise you will have fun.

### Lessons Learned
1. **Data Analysis** - I was pretty naive that the Public notebooks had enough data analysis in them.  I was amazed at how much information was publicly shared but you still need to go through everything in detail.  Next time I will make sure to spend a significant amount of time at the start on this.  I missed the issues with S and A in the 2017 data.
2. **CV/PLB/Submit Early** - I didn't full appreciate how important this is.  Without accurate metrics its easy to go with stuff that doesn't help and throw away stuff that does help.  I didn't submit to the LB until 3 days before the deadline.  That was a huge mistake.  My CV was way lower and I didn't have time to figure it out.
3. **Try Every Easy Idea** -  I convinced myself future time steps wouldn't help (too non-linear/random/players passing through each other) and never tried it.  It would have been easy to hack up some linear next steps,
4. **Don't Fiddle With Params** -  I wasted a lot time fiddling with NN parameters.  I think I spent weeks tinkering.
5. **Don't expect the model to figure out everything** -  I was amazed though at how well the Transformer did.  I think I should have tried to help it out a bit more especially with the limit of 4 hours of CPU time.
6.  **Code Competitions - Leave Time** - I ran out of time and am not thrilled with my final two submissions.  This was pretty tricky though with not having the 2019 data until after the submission.

### Solution
I basically went with a bunch of Transformer Encoder Layers with tied weights.  Spent most of the time fiddling with its configuration and inputs/outputs.  I missed the issues with 2017 data which in hindsight led to a lot of my difficulties in getting it to run correctly.  Because of this I also never figured out the issues with the CV.  My best submission had a **0.01190 5-fold random CV and a 0.01295 PLB**.

My best discovery was definitely **concatenating the raw runner inputs onto every player**.  Looks like this allowed the model to figure out many of the features that the other competitors discovered to be important.
  
.png?generation=1575034228981776&amp;alt=media)

oX and oY are cos(Orientation) and sin(Orientation), respectively.  Position and Season are embeddings. (# values, # dims)  Positions = ['RB', 'OL', 'QB', 'TE', 'WR', 'CB', 'DL', 'LB', 'SS'].  Looks like I should have worked on removing this feature though based on reading about better models.  I have my fingers crossed that by embedding the season the model can figure out the issues with 2017 data.  🙈 

I played around a lot with parameter sharing in the transformer layers.  I found that 2 sets of weights interlaced seemed to work the best:

```
for _ in range(self.n_layer // 2):
            players = self.encoder_layer(players)
            players = self.encoder_layer2(players)
```

Although the gains were minor compared to just tying all weights together as was shown in:
ALBERT - https://arxiv.org/pdf/1909.11942.pdf

Still not sure why adding FC Layers at the end didn't help.  Guessing it might have had to do with the issues with S and A in 2017 and overfitting to it.  It was cool though to see a 10K parameter network (16 dim) do pretty well at making predictions.

Also the 107 outputs were for -7 to 99 yard range.  Interestingly I found -5 to 21 came out with the same scores.  I wasted a lot of time on a "long run" predictor.  I could never get it to be even close to predicting long runs.

Thanks to the hosts and all the competitors!  Good luck everyone!
