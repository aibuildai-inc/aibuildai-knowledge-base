# Share your approach

Competition: battlefin-s-big-data-combine-forecasting-challenge
Rank: #23
Source: https://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/discussion/5966

<p>I am not in top 10. But I have used a lot from what others have done, especially from Miroslaw Horbal. So I want to share what I have done. Hopefully, those in top 10 could give us some thoughts. If those winners can share, that will be the best.&nbsp;</p>
<p>I have used two approaches:</p>
<p>1. use gbm to do feature selection. Then use linear regression with l1 loss, you can find Miroslaw's code:&nbsp;<a href="http://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/forums/t/5582/the-hint-thread">here</a>. This approach is simple. But it costs a lot of time. To do feature selection, it costs me around 24 hours in a server with 20 threads. But I have a server with 16 cores, 32 threads. So it's fine for me. I have uploaded the result of feature selection. You can download it directly, and put it into output dir.&nbsp;</p>
<p>2. use another model, can I say it as ar model. This also comes from Miroslaw's sharing:&nbsp;<a href="http://www.kaggle.com/c/battlefin-s-big-data-combine-forecasting-challenge/forums/t/5582/the-hint-thread?page=3">here</a>. I defined the method as this:</p>
<p>p = a0 * x0 + (1-a0)*a1*x1 + (1-a0)(1-a1)a2*x2 + ... + (1-a0)(1-a1)...(1-an-1) an xn + b.</p>
<p>To minimize the function, we have to define cost and gradient. You can refer to my code. This approach is quite efficient, it just costs me a few minutes. From private scores, this model is also better.&nbsp;</p>
<p>With the first approach, public score: 0.41820 private score: 0.42668</p>
<p>With the second approach, public score: 0.41833 private score: 0.42532</p>
<p>To run the code:</p>
<p>mkdir data, output, res</p>
<p>python2 dataProcess.py</p>
<p>python2 model linr se</p>
<p>python2 model ar</p>
<p>&nbsp;</p>
