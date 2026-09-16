# 24th Place Solution saintpp + lgb private0.810

Competition: riiid-test-answer-prediction
Rank: #24
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209659

Firstly, thanks to Kaggle hold such a good competition, almost no shake. Congrats tops and very thanks to my teammates @jaideepvalani  @ekffar. Finally our team finally public score 0.808, private 0.810, and final rank 24. we missed gold, it's ok, now i want to share our method

because of jet lag, i have no read discuss in detail, so i just see my teammate @jaideepvalani had shared our [solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209625) right now. As a complementary, not repeat, i just expand the detail based on  [jaideepvalani's solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209625)


# **Architecture**
    SaintPlusPlus, lgb
    

# Embeddings
 
## Exercise:
    Question_id
    Part
    Community id (tried, but no boosting, maybe need tuning)
    Task container id (tried, but no boosting, maybe need tuning)
## Interaction:
    lag time  (same container id, keep same with first one) (discrete embedding)
    prior question elapsed time(continuous embedding)
    prior had explain (discrete embedding)
    Prior lecture(discrete embedding)
## Response:
    Prior question correctness(discrete embedding)

# Data sample optimization
Data sampling is important in our experiment



# optimization
 ## SAINTPP
   1 skip connection, we add skip connection for multi encoder layers inner, and multi decoder layers inner, the origin implement i use from github have no this skip connection 
   2 add dropout for encoder, decoder, FFN
   3 try TCN causal conv after embedding, no boost, but now, i think it should tune and will work
   4 big seq len for saintpp give our team boosting, i think it maybe caused by too short seq make 
       transformer link model struggle into local minest point 

## LGB
updating

# Conclusion

Our best saintpp(seqlen 320, embdim224, 2 x encder+ 2 x decoder) get cv 0.8016 lb 0.806, and lgb near get 0.796. Finally submission is merge lgb(lb 796) + saintpp cv1(cv 8016) + saintpp cv3(cv 0.7992)  = public lb 0.808, and private 0.810.
should do things:
  1 should continue optimize our net with casual conv(because transformer like is a global attention, some local correlation maybe ignored) and gru, but finally have no time
  2 should try big net continue

Thanks again to my teammates, they give me a lots of help and let me learn a lot. Going  for gold next Compete, keep moving.
