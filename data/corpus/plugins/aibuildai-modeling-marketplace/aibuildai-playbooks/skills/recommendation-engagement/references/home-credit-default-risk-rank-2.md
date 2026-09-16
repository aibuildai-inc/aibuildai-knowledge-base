# 2nd place solution ( team ikiri_DS )

Competition: home-credit-default-risk
Rank: #2
Source: https://www.kaggle.com/c/home-credit-default-risk/discussion/64722

Congratulations to the winner Home Aloan and to all participants in this competition!

This competition was very tough for us and the biggest competition ever.   
Thank you Home Credit and kaggle for organizing this great competition.

ikiri_DS consists of 12 kagglers and we can not write up detail of each solution at once.  
So we decided to post our model pipeline and short description of individual approaches at first.  
We will add more descriptions with requests in your comments of this discussion.  
Please feel free to ask us questions and comments, thanks.  
  
  
----------

**Member comments**  
  
 - **[ONODERA][1]**  
I'm in charge of feature engineering. I don't have any my models.
Here is my [git][2].  
  
 - **[RK][3]**  
Feature  
・Various dimension reductions (PCA, UMAP, T-SNE, LDA...)  
・Genetic programming application (from kernel)  
・Brute force feature search from ~1TB feature pools  
Model  
・Algorithm comparison (Catboost, LightGBM with dart, and so on)  
・Parameter tuning  
Blending  
・Algorithm comparison (Direct AUC maximization with modified Powell algorithm is good for making my 2nd level model)  
  
 - **[Yuya Yamamoto][4]**  
Interest rate feature, finding about difference between train and test data and residual correction of Neural Network predictions.  I have posted my finding and others in [https://www.kaggle.com/c/home-credit-default-risk/discussion/64784][5].  
  
 - **[tosh][6]**  
I'm in charge of Neural Network(NN) in this competiton.
[https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#380224][7]
  
 - **[ireko8][8]**  
DAE based on porto seguro 1st solution  
[https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#379891][9]  
  
 - **[tereka][10]**  
CNN and RNN models.  
[https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#379880][11]  
  
 - **[branden][12]**  
  
 - **[takuoko][13]**  
I worked with angus and branden.   
Our task in team ikiri_DS is to generate diversity, and we could do it!   
I made LGBM ( private 0.79966 / public 0.80328 ).  
  
 - **[Angus][14]**  
Hello ~ I am Shuo-Jen, Chang - a kaggler from Taiwan. you can just call me Angus. My job in ikiri_DS is to generate diversity by modeling with my feature set. see the attachment below ( HC - Brief solution from Shuo-Jen, Chang.html )!
  
 - **[Tam][15]**  
  
 - **[Giba][16]**  
Post Processing  
discussion  
[https://www.kaggle.com/c/home-credit-default-risk/discussion/64485][17]  
kernel  
[https://www.kaggle.com/titericz/giba-post-processing-user-id-boost][18]  
  
 - **Maxwell**  
Meta features on train_app and bureau( gain : public +0.002 / private +0.003 ), some models( LGBM, ExtraTree ) for model diversity and blending using adversarial validation( gain : private +0.0002 )  
  
![model pipeline of team ikiri_DS][19]


  [1]: https://www.kaggle.com/onodera
  [2]: https://github.com/KazukiOnodera/Home-Credit-Default-Risk
  [3]: https://www.kaggle.com/ryuichi0704
  [4]: https://www.kaggle.com/nejumi
  [5]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64784
  [6]: https://www.kaggle.com/ktts1031
  [7]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#380224
  [8]: https://www.kaggle.com/ireko8
  [9]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#379891
  [10]: https://www.kaggle.com/tereka
  [11]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64722#379880
  [12]: https://www.kaggle.com/brandenkmurray
  [13]: https://www.kaggle.com/takuok
  [14]: https://www.kaggle.com/andrew60909
  [15]: https://www.kaggle.com/nthanhtam
  [16]: https://www.kaggle.com/titericz
  [17]: https://www.kaggle.com/c/home-credit-default-risk/discussion/64485
  [18]: https://www.kaggle.com/titericz/giba-post-processing-user-id-boost
  [19]: https://cdn-ak.f.st-hatena.com/images/fotolife/g/greenwind120170/20180901/20180901083809.png
