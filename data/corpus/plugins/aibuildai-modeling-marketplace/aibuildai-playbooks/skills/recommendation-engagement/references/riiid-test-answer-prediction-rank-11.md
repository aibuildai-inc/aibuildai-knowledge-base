# 11th Place Solution

Competition: riiid-test-answer-prediction
Rank: #11
Source: https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/209694

Thanks to my teammate [Akihiko](https://www.kaggle.com/jusco11) for competing with me on this, there was great learning from the community for both of us. And thanks to our hosts for a wonderful challenge. Congrats all who competed. 

Solution has heavily inspired by Bestfitting's [TalkingData solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262) which we noticed a lot of the people in this competition took part in :) 

We used around 35 features including the raw, in a 2 LSTM layer model, each layer single direction trained on sequence length 256, infer on 512. 
Batchsize for training 2048. Hidden layer size of 512. 
Example below.  

First layer,
- Used features below in `embcatq`, no continuous features. 
- No label in first layer, like in the SAINT paper.
- Added the difference of some of the embedding to the final embedding. This gives the model info on how similar each historical question was to the question in the sample.   

Second layer, 
- Outputs of first layer and included continuous features. 
- Added embedding for interaction of question and chosen answer. 
- Continuous features generated using @its7171 great book - this is why I would like to be able to give multiple upvotes.
- One important feature was the answer ratio, what percentage of students picked the same answer as the chosen answer.   

How to handle the histories in memory was a problem, but there was plenty of space for it on the GPU, so loaded that first to numpy then to a torch tensor on GPU (history features took around 6GB); then loaded the other objects to RAM. 
Attention did not work for us - looked promising on validation though - should have persisted with it.  We just took the final hidden cell from the LSTM as output.  
The below got ~ 0.811 public, by make some changed to the architecture and bagging four models, lifted to 0.813 public, 0.816 private. 


```
class LearnNet(nn.Module):
    def __init__(self, modcols, contcols, padvals, extracols, 
                 dropout = 0.2, hidden = args.hidden):
        super(LearnNet, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.modcols = modcols + extracols
        self.contcols = contcols
        
        self.emb_content_id = nn.Embedding(13526, 32)
        self.emb_content_id_prior = nn.Embedding(13526*3, 32)
        self.emb_bundle_id = nn.Embedding(13526, 32)
        self.emb_part = nn.Embedding(9, 4)
        self.emb_tag= nn.Embedding(190, 8)
        self.emb_lpart = nn.Embedding(9, 4)
        self.emb_prior = nn.Embedding(3, 2)
        self.emb_ltag= nn.Embedding(190, 16)
        self.emb_lag_time = nn.Embedding(301, 16)
        self.emb_elapsed_time = nn.Embedding(301, 16)
        self.emb_cont_user_answer = nn.Embedding(13526 * 4, 5)
            
        self.tag_idx = torch.tensor(['tag' in i for i in self.modcols])
        self.cont_wts = nn.Parameter( torch.ones(len(self.contcols)) )
        self.cont_wts.requires_grad = True
        self.cont_idx = [self.modcols.index(c) for c in self.contcols]
        
        self.embedding_dropout = SpatialDropout(dropout)
        
        self.diffsize = self.emb_content_id.embedding_dim + self.emb_part.embedding_dim + \
                        self.emb_bundle_id.embedding_dim + self.emb_tag.embedding_dim * 7 
        IN_UNITSQ = self.diffsize * 2 + \
                    self.emb_lpart.embedding_dim + self.emb_ltag.embedding_dim + \
                        self.emb_prior.embedding_dim + self.emb_content_id_prior.embedding_dim + \
                        len(self.cont_idxcts)
        IN_UNITSQA = ( self.emb_lag_time.embedding_dim + self.emb_elapsed_time.embedding_dim + \
                self.emb_cont_user_answer.embedding_dim) + len(self.contcols)
        LSTM_UNITS = hidden 
        self.diffsize = self.emb_content_id.embedding_dim + self.emb_part.embedding_dim + \
                        self.emb_bundle_id.embedding_dim + self.emb_tag.embedding_dim * 7 
        
        self.seqnet1 = nn.LSTM(IN_UNITSQ, LSTM_UNITS, bidirectional=False, batch_first=True)
        self.seqnet2 = nn.LSTM(IN_UNITSQA + LSTM_UNITS, LSTM_UNITS, bidirectional=False, batch_first=True)
            
        self.linear1 = nn.Linear(LSTM_UNITS * 2 + len(self.contcols), LSTM_UNITS//2)
        self.bn0 = nn.BatchNorm1d(num_features=len(self.contcols))
        self.bn1 = nn.BatchNorm1d(num_features=LSTM_UNITS * 2 + len(self.contcols))
        self.bn2 = nn.BatchNorm1d(num_features=LSTM_UNITS//2)
        
        self.linear_out = nn.Linear(LSTM_UNITS//2, 1)

        
    def forward(self, x, m = None):
        
        ## Continuous
        contmat  = x[:,:, self.cont_idx]
        contmat = self.bn0(contmat.permute(0,2,1)) .permute(0,2,1)
        contmat = contmat * self.cont_wts
        
        content_id_prior = x[:,:,self.modcols.index('content_id')] * 3 + \
                            x[:,:, self.modcols.index('prior_question_had_explanation')]
        embcatq = torch.cat([
            self.emb_content_id(x[:,:, self.modcols.index('content_id')].long()),
            self.emb_part(x[:,:, self.modcols.index('part')].long()), 
            self.emb_bundle_id(x[:,:, self.modcols.index('bundle_id')].long()),
            self.emb_tag(x[:,:, self.tag_idx].long()).view(x.shape[0], x.shape[1], -1),
            self.emb_prior(x[:,:, self.modcols.index('prior_question_had_explanation')].long() ),
            self.emb_lpart(x[:,:, self.modcols.index('lecture_part')].long()), 
            self.emb_ltag(x[:,:, self.modcols.index('lecture_tag')].long()) , 
            self.emb_content_id_prior(  content_id_prior.long()),
            ], 2)
        embcatqdiff = embcatq[:,:,:self.diffsize] - embcatq[:,-1,:self.diffsize].unsqueeze(1)
            
        # Categroical embeddings
        embcatqa = torch.cat([
            self.emb_cont_user_answer(x[:,:, self.modcols.index('content_user_answer')].long()),
            self.emb_lag_time(x[:,:, self.modcols.index('lag_time_cat')].long()), 
            self.emb_elapsed_time(x[:,:,self.modcols.index('elapsed_time_cat')].long())
            ] , 2)
        #embcatqadiff = embcatqa - embcatqa[:,-1].unsqueeze(1)
        embcatq = self.embedding_dropout(embcatq)
        embcatqa = self.embedding_dropout(embcatqa)
        embcatqdiff = self.embedding_dropout(embcatqdiff)
        
        # Weighted sum of tags - hopefully good weights are learnt
        xinpq = torch.cat([embcatq, embcatqdiff], 2)
        hiddenq, _ = self.seqnet1(xinpq)
        xinpqa = torch.cat([embcatqa, contmat, hiddenq], 2)
        hiddenqa, _ = self.seqnet2(xinpqa)
        
        # Take last hidden unit
        hidden = torch.cat([hiddenqa[:,-1,:], hiddenq[:,-1,:], contmat[:, -1]], 1)
        hidden = self.dropout( self.bn1( hidden) )
        hidden  = F.relu(self.linear1(hidden))
        hidden = self.dropout(self.bn2(hidden))
        out = self.linear_out(hidden).flatten()
        
        return out
```
