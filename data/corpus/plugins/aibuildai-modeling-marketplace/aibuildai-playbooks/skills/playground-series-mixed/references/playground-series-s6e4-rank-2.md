# 2nd Place - Claude Code and Codex - GPU LogReg

Competition: playground-series-s6e4
Rank: #2
Source: https://www.kaggle.com/c/playground-series-s6e4/writeups/2nd-place-claude-code-and-codex-gpu-logreg

Congratulations to all the winners and everyone who pushed hard. Kaggle's April Playground (predicting irrigation) competition was similar to Kaggle's March Playground (predicting churn) competiton [here][1]. 

April was 3-class multi-class classification and March was binary classification, so both were classification. The data was similar. Both competitions came from original data with minimal feature interactions (i.e. xgb optimal max depth on orginal data was max_depth=1 in both). Both comps had original data roughly 10k rows. And both comps had similar number of features with no NAN and some numeric and some categorical with similar cardinalities.

# Convert March Solution to April Solution
On the first day of the competition, I asked Claude Code cli to read all my local March solution Python scripts and convert them to April competition. Claude rewrote the 150 model scripts that I used and converted March to April. I ran the 150 scripts on GPU over the weekend and ensembled them. On the third day of the competition I had a private LB first place solution with `private LB 0.98160, public LB 0.98182, cv 0.98130`. The LLM Agent was so fast, there wasn't anything else to do for the remaining 27 days!

# Final Solution
After Claude Code cli converted my March solution to April and obtained `private LB 0.98160, public LB 0.98182`, there wasn't anything to do for 27 days. I didn't think I could improve because my March solution was already the result of LLMs working for 30 days and writing 600,000 lines of code. I also believed that the April and March comps were so similar that there wasn't any new signal to be found. None-the-less I built some more models and tried various things. Specifically I had Codex cli convert my March solution to April also.

My final submission was Claude Code's conversion ensemble blended with Codex's conversation ensemble! This solution achieved `private LB 0.98151, public LB 0.98195, cv 0.98170`. Unfortunately working for more days past day 3 improved CV score but decreased private LB by about 1e-4 :-(

# GPU Logistic Regression Stacker
One important aspect of my final solution is the ensemble technique. In this competition, the metric is `balanced accuracy`, so we want the stacker to be true multinomial logistic regression with weighted classes. We also need it to run fast, so we can repeatedly ensemble potentially 100s of models. And do forward or backward feature selection if desired. Unfortunately NVIDIA cuML's logistic regression (which we used in March comp) isn't multinomial and doesn't support class weighting. So Claude wrote me a fast GPU multinomial logistic regression using PyTorch

    class PyTorchMultiLogReg(nn.Module):
        def __init__(self, in_features, out_features=3):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

It is often overlooked but logistic regression is just an NN with input going to output (and no middle hidden layers). Additionally to mimic logistic regression with L2 regularization we need weight decay on the coefficients but not the bias

        criterion = nn.CrossEntropyLoss(reduction='none')
        weight_decay_mapped = 1.0 / (C * len(y_tr))
        optimizer = optim.Adam([
            {'params': model.linear.weight, 'weight_decay': weight_decay_mapped},
            {'params': model.linear.bias, 'weight_decay': 0.0}
        ], lr=0.01)

And we want class weights

        classes = np.unique(y_tr)
        cw = compute_class_weight('balanced', classes=classes, y=y_tr)
        dd = dict(zip(classes, cw))
        sample_weights = np.array([dd[c] for c in y_tr], dtype=np.float32)
        sw_t    = torch.tensor(sample_weights, dtype=torch.float32, device=device)
        loss = (criterion(logits, y_tr_t) * sw_t).mean()

# CV to Private LB Correlation
Unfortunately for me in this competition, the metric and CV to Private LB correlation was a little noisy (compared to how close participants were clustered on the leaderboard). In the image below, we can see that I had many ensembles with similar CV = 0.98175. However those multiple submission.csv files had Private LB ranging from 0.98130 to 0.98170. My best unselected submission is `CV = 0.98175, Private LB = 0.98172, Public LB = 0.98182`. 


# CV to Public LB Correlation


# Solution Code
I will be publishing my final submission code which includes GPU PyTorch multinomial logistic regression soon...


[1]: https://www.kaggle.com/competitions/playground-series-s6e3
[2]: https://www.kaggle.com/competitions/playground-series-s6e4/writeups/1st-place-one-vs-rest-approach
