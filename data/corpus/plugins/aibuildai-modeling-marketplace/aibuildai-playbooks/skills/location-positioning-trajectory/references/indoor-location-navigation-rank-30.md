# Part of the 30th place solution

Competition: indoor-location-navigation
Rank: #30
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240056

Congratulations to all the winners! 
It was actually a tough competition but we enjoyed and learned a lot!
I want to say thank you to all the kagglers who participated in the comp! 

Our team got 30th place, and actually my model was not so important for our result.
Team mates' models perform far better than mine.
But our team agreed that we won't publish our solution. 
Still, I think I want (or need) to publish some of my work so I posted it. 

## The LSMT notebook
I published [the LSTM notebook](https://www.kaggle.com/kokitanisaka/lstm-by-keras-with-unified-wi-fi-feats) 2 months ago, and I want to show how it went after that. 

[This is the notebook. ](https://www.kaggle.com/kokitanisaka/self-attentintive-lstm-by-keras)

I applied self-attention layer and some more modifications so it got better. 
Actually the performance of the notebook is so so. Public score is 6.060. 
Feel free to throw any comments or feedback. Thanks!

## Post process
I tried a post process but it didn't work for public LB so we didn't use it.
But when I see the private score, it seems it worked. 
So I choose to publish the following notebook as well.

The idea is to fix the result of the snap to grid. 
We thought that the ground truth should be on grids, so we relied on snap to grid.
But after we applied snap to grid, some paths looked definitely wrong, so I tackled the issue. 

[This is the preparation for the pp.](https://www.kaggle.com/kokitanisaka/create-arrayed-map)
[And this is the pp.](https://www.kaggle.com/kokitanisaka/fix-snapped-waypoints)

Again, thank you to all of you! I got another memorable competition that I joined. I really enjoyed it! 
Look forward to seeing you in other competitions!
