# 1st Contest- 15th Place

Competition: playground-series-s6e1
Rank: #15
Source: https://www.kaggle.com/c/playground-series-s6e1/writeups/1st-contest-15th-place

I hoped I'd score a top 20% position for my 1st contest, like 500th at best, but I got extremely lucky and secured a top 20 score ! 

At first I tried to do as much feature engineering as possible to inject in my Random forest, I stacked over 250 features, from the columns squared, to the individual digits of the study hours and their combinations. This made me reach the 550th place which was my initial goal.

And then I read @include4eto notebook and it taught me about meta learning and putting a group of pretrained models in a linear regression can be useful. I copied his code which had around 58 pretrained models, and this improved my accuracy a bit.

But what was interesting about his middle, maximum, and minimum approach was the idea of predicting a region, not a value. I took it and further improved it, where I predicted one of 7 quantile regions, a model that had an accuracy of 0.48, at first it was useless, but when I used it to improve his 58 pretrained models gating them (using their mean, and standard deviation) according to the regions predicted probability, I had one of the happiest moments of my life: I jumped from 500th to 9th on the public leaderboard!!

The code I used : 
`


    def soft_gate_preds_conf(base_pred, proba, anchors, alpha=0.30, temp=1.0, conf_pow=1.8):
    p = np.clip(proba, 1e-6, 1.0).astype(np.float32)

    # temperature
    p = p ** (1.0 / float(temp))
    p = p / p.sum(axis=1, keepdims=True)

    conf = p.max(axis=1)
    w = (conf ** float(conf_pow)).astype(np.float32)

    gated_target = (p @ anchors).astype(np.float32)
    out = base_pred + (float(alpha) * w) * (gated_target - base_pred)
    mean_expert_oof  = X_train_hc.mean(axis=1)
    std_expert_oof   = X_train_hc.std(axis=1) + 1e-6

    mean_expert_test = X_test_hc.mean(axis=1)
    std_expert_test  = X_test_hc.std(axis=1) + 1e-6`
    # weight: more trust to gated stack when experts agree (low std) AND classifier confident
    conf_oof  = oof_pred_cf_proba.max(axis=1)
    conf_test = test_proba_avg.max(axis=1)

    w_oof  = (conf_oof / (1.0 + std_expert_oof)).astype(np.float32)
     w_test = (conf_test / (1.0 + std_expert_test)).astype(np.float32)


    # normalize weights into [0,1]
    w_oof  = (w_oof - w_oof.min()) / (w_oof.max() - w_oof.min() + 1e-6)
    w_test = (w_test - w_test.min()) / (w_test.max() - w_test.min() + 1e-6)

    oof_blend  = w_oof * best_oof_gated  + (1 - w_oof) * base_oof
    test_blend = w_test * best_test_gated + (1 - w_test) * base_test




Unfortuanately, I didn't know more models would've been more beneficial as I saw the other writeups trained up to 200, also I forgot about the idea of hyperparameter tuning for the gates and the model (don't judge it's my 1st competition afterall).

Overall, I am extremely excited for the next playground to apply what I learned during this challenge, hopefully it turns out to be more than luck and I manage to get a top 20 rank again, and I thank everyone for the wonderful discussions and notebooks, really I learned a ton from every one here!
