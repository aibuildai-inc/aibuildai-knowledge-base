# 4th place - 5 day rush

Competition: playground-series-s6e5
Rank: #4
Source: https://www.kaggle.com/c/playground-series-s6e5/writeups/4th-place-5-day-rush

Wow! I restarted on this competition with only 5 days remaining (at 0.95406 LB) and I only expected a top 10 finish at most. Turns out its a top 4 finish :D! As usual, many thanks to the people who shared their code and insights, including but not limited to: @yekenot, @masayakawamata, @mikhailnaumov, @ern711, @onurkoc92.

Now onto my solution (which is fairly standard :V):

#FE

For GBDTs, my best feature set was relatively small:

- Count encoding on categorical columns
- Arithmetic interactions from @yekenot's (overpowered) RealMLP notebook
- TE on some pairs (also from @yekenot's notebook)

I didn't really train that many NNs myself, so I just used the same feature set to train them, mostly just yoinked the oofs from various public NNs, they did the job

#Models

I only really want to list out my best XGB's scores: CV 0.95378, Public 0.95313, Private 0.95346. If I listed out other model's scores, they would all be models from public notebooks :V

#Ensembling

I tried a few ensemblers as usual and HC was the best for me, so I used it to the end. The final ensemble has CV 0.95529, Public 0.95469, Private 0.95488. I blended it with @mikhailnaumov's ensemble at the end since I couldn't reproduce his scores, and got Public 0.95471 Private 0.95490. Without his ensemble, I would have landed 6th :V

#Conclusion

In conclusion, trust in yourself, even when you're ranked ~400th place in a comp with 5 days left. Happy Kaggling!
