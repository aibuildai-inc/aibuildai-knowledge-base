# A CNN classifier and a Metric Learning model,1st Place Solution

Competition: human-protein-atlas-image-classification
Rank: #1
Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109

Congrats to all the winners, and thanks to the host and kaggle hosted such an interesting competetion.<br>
I am sorry for late share, I have worked hard to prepare it in recent days,tried to verify my solution and to make sure it’s reproducible,stable,efficient as well as interpretative.

**Overview**
![enter image description here][1]
**Challenges:**<br>
*Extreme Imbalance,rare classes hard to train and predict but play an important role in the score.*<br>
*Data distribution is not consistent in train set,test set,and HPA v18 external data.*<br>
*The images are with high quality,but we must find a balance between model efficiency and accuracy.*<br>

**Validation for CNNs:**<br>
I split the val set according to https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/67819 great thanks to @trentb

I found Focal Loss of the whole val set is a relative good metric to the model capability, F1 is not a good metric as it’s sensitive to the threshold and the threshold is depend on the distribution of the train and val set.

I tried to evaluate the capability of a model by set the ratio of each class to the same as train set. I did so because I thought I should not adjust the thresholds according to the public LB,but if I set the ratio of the prediction stable,and,if the model is stronger,the score will improve. That’s to say,I used public LB as another validation set.

**Training Time Augmentations:**<br>
Rotate 90,flip and randomly crop 512x512 patches from 768x768 images(or crop 1024x1024 patches from 1536x1536 images)


**Data Pre-Processing:**
Remove about 6000 duplicates samples from v18 external data, using hash method which been used to find test set leak.

Calculate mean and std using train+test,and used them before feeding images to the model.

**Model training:**<br>
**Optimizer**:Adam<br>
**Scheduler**:<pre>lr = 30e-5
if epoch &gt; 25:
    lr = 15e-5
if epoch &gt; 30:
    lr = 7.5e-5
if epoch &gt; 35:
    lr = 3e-5
if epoch &gt; 40:
    lr = 1e-5
</pre>
**Loss Functions**:FocalLoss+Lovasz,I did not use macro F1 soft loss, because the batch size is small and some classes are rare, I think it’s not suitable for this competition.I used lovasz loss function because I thought although the IOU and F1 are not the same,but it can balance the Recall and Precision to some extend.

**I did not use oversample.**<br>
**Model structure:**
My best model is a densenet121 model, which is very simple,the head of the model is almost same as public kernel https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb by @iafoss.
<pre>  (1): AdaptiveConcatPool2d(
    (ap): AdaptiveAvgPool2d(output_size=(1, 1))
    (mp): AdaptiveMaxPool2d(output_size=(1, 1))
  )
  (2): Flatten()
  (3): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (4): Dropout(p=0.5)
  (5): Linear(in_features=2048, out_features=1024, bias=True)
  (6): ReLU()
  (7): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (8): Dropout(p=0.5)
  (9): Linear(in_features=1024, out_features=28, bias=True)
</pre>
I tried all kinds of network structure according to the Multi-Label classification papers, the results were not improved instead of their beautiful structures and theory behind them.:) <br>
**Prediction time augmentations:**
I predicted the test set by using best focal loss epoch with 4 seeds to random crop 512x512 patches from 768x768 images, and got the max probs from the predictions. 

**Post-processing:**
At the final stage of the competition, I decided to generate two submissions:
1.The first one was keep the ratio of the labels to the public test set,since we did not know the ratio of the rare classes,I set them to the ratio of the train set.
2.The second one was keep the ratio of the labels to the average ratio of train set and public test set.

Why? Although I tried to add or reduce the count of rare classes by 2-5 samples,the public LB can improve, but this was a dangerous way.I just only used it to   evaluate the possible shakeup.
<hr>
Metric Learning:
I took part in the landmark recognition challenge in May 2018,https://www.kaggle.com/c/landmark-recognition-challenge,and I had planed to use metric learning in that competition,but time was limited after I finished the TalkingData competition. But I read many  papers related,and did many experiments after that. 

When I analyzed the predictions of my models,I wanted to find the nearest samples to compare,I first used the features from CNN model,I found they are not so good,so I decided to try Metric Learning.

I found it’s very hard to train in this competition,it took me a lot of time but the result was not so good,and I found the same algorithm can work very well in Whale identification competition,but I did not give up and finally found a good model in last two days.

By using the model,I could find the nearest sample on validation set,**the top1 accuracy &gt;0.9**
These are the demo:<br>
Correct sample with single Label
![][2]
Correct sample multiple labels
![][3]
Correct sample with rare label:Lipid droplets
![][4]
Correct sample with rare label:Rods &amp; rings
![][5]
Missed a label
![][6]
Incorrectly add a label
![][7]

Since the top1 accuracy&gt;0.9,I thought I could just use the metric learning result to set the labels of test set. But I found that the test set is a little different to V18, and some of samples can not find nearest neighbor in train set and V18. So I set a threshold and replace the labels with found sample’s. Fortunately,the threshold is not sensitive to the threshold. Replacing 1000 samples in test set is almost the same score as replacing 1300 samples. By doing so, my score can improve 0.03+,which was a huge improvement in this competition.

I think my method is important not only improve the score,it can help HPA and their users in following way:<br>
*1.When someone want to label or learn to label an image or check the quality, he can get the nearest images for referring to.<br>
2.We can cluster the images by the metric and find the label noises and then improve the quality of the labels.<br>
3.We can explain why the model is good by visualizing the predictions.<br>*

**Ensemble:**
To keep the solution simple,I don't discuss the ensemble here, a single model or even a single fold + metric learning result is good enough to get the first place.

**The scores on LB:**
![][8]
<br>
I am sorry I can not describe the details of this part now, as I mentioned before,the whale identification competition is still on-going. 


**Introspection**:
Before I entered this competition,I never expected I can find a way out,it’s very hard to build a stable CV and the score is sensitive to the distribution of rare classes.A gold medal is my max expectation.<br>
I feel kaggle competitions are becoming harder and harder.In all honesty,there are no secrets but hard work.I treat every competition as a force to push me forward.I force myself not to learn and use too much competition skills but knowledge to solve real problems.<br>
It’s quite lucky I found a relatively good solution in this competition as I failed to find a Reinforcement Learning algorithm in Track ML, and failed to finish a good CNN-RNN model in Quick Draw competition in time,but anyway,if we compete only for win,we may loss,if we compete for learning and providing useful solution to the host,nothing to loss.

**Update,Metric Learning Part:**

Sorry for late update!

As I noticed that the sample with same antibody-id have almost same labels, so I thought I may treat the antibody-id as face id, and use face-recognition algorithms on HPA v18 dataset.

When training, I used V18 data antibody IDs to split the samples,keep a sample in validation set,and put the other samples with same ID in train set.I used top1-acc as validation metric.

**Metric Learning Model:**
Network:resnet50
Augmentations:Rotate 90,flip
Loss Functions:ArcFaceLoss
Optimizer:Adam
Scheduler:lr = 10e-5, 50 epochs.

**Model details:**

<pre>class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine &gt; 0, phi, cosine)
        else:
            phi = torch.where(cosine &gt; self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine

 def __init__(self,....
 	... ...
	self.avgpool = nn.AdaptiveAvgPool2d(1)
	self.arc_margin_product=ArcMarginProduct(512, num_classes)
	self.bn1 = nn.BatchNorm1d(1024 * self.EX)
	self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
	self.bn2 = nn.BatchNorm1d(512 * self.EX)
	self.relu = nn.ReLU(inplace=True)
	self.fc2 = nn.Linear(512 * self.EX, 512)
	self.bn3 = nn.BatchNorm1d(512)

def forward(self, x):
	... ...
	x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
	x = x.view(x.size(0), -1)
	x = self.bn1(x)
	x = F.dropout(x, p=0.25)
	x = self.fc1(x)
	x = self.relu(x)
	x = self.bn2(x)
	x = F.dropout(x, p=0.5)

	x = x.view(x.size(0), -1)

	x = self.fc2(x)
	feature = self.bn3(x)

	cosine=self.arc_margin_product(feature)
	if self.extract_feature:
	    return cosine, feature
	else:
	    return cosine
</pre>

Please refer to the paper:
ArcFace: Additive Angular Margin Loss for Deep Face Recognition 
https://arxiv.org/pdf/1801.07698v1.pdf
Deep Face Recognition: A Survey    
https://arxiv.org/pdf/1804.06655.pdf

As I was very busy after this competition(and will be for a little long time),I used almost the same model finsihed the Whale competition and the winners' models are very good, so I think I need not write a summary of that competition. I the person re-idenfication related papers and solutions are good choice to Whale competition.

Thanks for your patience! 


  [1]: https://bestfitting.github.io/kaggle/protein/images/001_pipeline.png
[2]:https://bestfitting.github.io/kaggle/protein/images/002_Sample%20with%20single%20Label.jpg
[3]:https://bestfitting.github.io/kaggle/protein/images/003_Sampe%20with%20multi%20labels.jpg
[4]:https://bestfitting.github.io/kaggle/protein/images/004_Rare%20Label_Lipid%20droplets.jpg
[5]:https://bestfitting.github.io/kaggle/protein/images/005_Rare%20Label_Rods%20%26%20rings.jpg
[6]:https://bestfitting.github.io/kaggle/protein/images/006_Missed%20a%20label.jpg
[7]:https://bestfitting.github.io/kaggle/protein/images/007_incorrectly%20add%20a%20label.jpg
[8]: https://bestfitting.github.io/kaggle/protein/images/008_scores.png
