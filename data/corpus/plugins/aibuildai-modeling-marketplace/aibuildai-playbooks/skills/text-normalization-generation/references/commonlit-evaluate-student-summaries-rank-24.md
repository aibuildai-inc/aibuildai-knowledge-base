# Public 5th & Private 24th Place Solution (CV = LB)

Competition: commonlit-evaluate-student-summaries
Rank: #24
Source: https://www.kaggle.com/c/commonlit-evaluate-student-summaries/discussion/446689

Congratulations to the winners and everyone who put lots of efforts on this competition, and thank Kaggle & host for organising this competition.
Even though my team shakedown on the private learderboard, we don't think it's truly 'shakedown'. Because our CVs are very close to the LBs (public + private) and we have already selected almost our best results. Our final solutions are purely based on best CVs.

# 1 Summary
## 1.1 Key points
●	Various inputs using common words, phrases, etc. between “prompt_text” and “text” as inputs.
●	Custom heads combining various pooling.
●	Pseudo labeling with back translated texts.
●	Weighted loss.
●	Ensemble weights based on ridge regression with large alpha 
## 1.2 Our submissions (3 final submissions and the best submission)
	CV	Public LB	Private LB
①best public LB	0.4596	0.420	0.463
②best CV	0.4566	0.423	0.463
③other	0.4579	0.422	0.464
(best private LB)	0.4601	0.420	0.462
cv roughly euqals to LB: 0.457 = 0.420x0.13(public) + 0.463x0.87(private)

# 2 Details
## 2.1 Various inputs
We added common words, phrases, etc. between “prompt_text” and “text” to the end of the input texts.(While 2 and 3 grams overlap were added to the input, 2nd Stage LGB models are not working anymore, at least this is the case on Jie’s models)
●	overlap words
●	overlap N-grams (N=2,3)
●	overlap phrases
●	overlap key phrases by Rake method
 
 
 
## 2.2 Custom heads
We created many custom heads that cat the output values of different poolings.
●	CLS output
●	Mean pooling
●	Max pooling
●	GeM pooling
●	Attention pooling
●	Mean-Max-Attention Pooling (MMA)
●	Mean-Max-Gem Pooling (MMG)
●	Gem-Attention Pooling (GA)
Example: GA Pooling
gemtext_out = self.gempooler(lasthiddenstate, attention_mask)
attpool_out = self.attpooler(lasthiddenstate, attention_mask)
context_vector = torch.cat((gemtext_out, attpool_out), dim=-1)

## 2.3 Pseudo labeling for back translated texts.
Data Augmentation and Pseudo Labelling: Translate summary text to Chinese, then translate back to English, then use a good model to Pseudo label.
## 2.4 Weighted loss
We created a weighted loss for wording and content as following, ratios are decided by the RMSE ratio between content and wording.
class WeightedCRMSE(nn.Module):
    def __init__(self):
        super(WeightedCRMSE, self).__init__()
        self.weights = torch.tensor([0.85, 1.15]).to(device)

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_root_squared_errors = torch.mean(squared_errors, dim=0) ** 0.5 * self.weights
        loss = torch.mean(weighted_root_squared_errors)
        return loss
## 2.5 Ensemble weights based on ridge regression with large alpha
We trained ridge regression out-of-fold and use the mean of the normalized coefficients as the ensemble weights. Both CV and LB scores were better when using ridge regression with a large alpha than using linear regression and other methods.
1.	The intercept is zero.
2.	Large alpha (500)

## 2.6 Others
●	Backbone
○	deberta-v3-large
○	electra-large
○	roberta-large
○	roberta-large-squad2
●	Model for ensemble
○	LightGBM
○	Ridge regression
●	Max length
○	300
○	512
○	768

# 3 Not working
●	LSTM head
●	Freezing the embbedings and some layers.
●	AWP
●	MLM
●	30° rotated targets
●	Different max length between training and inference.


# 4 Jie’s model performance
●	Name convention backbon_maxlength_2&3gramsoverlap_dataaugmentionflag_pooling, e.g.
del_768_23_aug_mmg: Debertal-V3-Large, maxlength=768, 2and3 grams overlap were added to input, data augmention is used for model training, use Mean-Max-Gem pooling.
●	Table below: 1st row: model name, 2nd row: CV(average of 4 folds), 3rd row: CV (oof), 4th row: LB, 5th row: Inference time. 
●	Ensemble of LB 0.437, 0.432, 0.439, 0.436, 0.433 and 0.436 gives CV: 0.45909, Pubic LB 0.424, Private LB: 0.464
●	Note: Deberta Large models inference time are 80mins not 125-140min, all other models are <=36mins



# 5 yyykrk’s model performance
 
