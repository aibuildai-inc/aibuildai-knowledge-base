# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #11
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023#44090

<p>This is my first Kaggle competition and I really enjoyed playing it. Thanks for everyone who shared your innovative&nbsp;models! Here is mine:</p>

<p>( notation: s<strong>x</strong>d<strong>y&nbsp;</strong>means store <strong>x</strong>'s department <strong>y</strong>)</p>
<p>1.</p>
<p>Impute all missing values by their corresponding highly correlated s<strong>i</strong>d<strong>j</strong>. For example, there is a&nbsp;missing value in the 80th week in s1d1, and s2d1 is highly correlated with s1d1, then&nbsp;the algorithm uses the formula: average(s1d1 / s2d1) * 80th week in s2d1 to estimate 80th week in s1d1. If there are only a few non-missing data in a specific s<strong>i</strong>d<strong>j</strong>, its average value will be estimated using other averages of departments by stepwise&nbsp;regression / regularization, then impute with&nbsp;a highly correlated s<strong>i</strong>d<strong>j</strong>&nbsp;just like above. &nbsp;</p>
<p>2.&nbsp;</p>
<p>For every s<strong>i</strong>d<strong>j</strong>, use &quot;time series CV&quot; to choose the best two models from different variations of stl&nbsp;decomposition + arima/ets. According to Dr. Hyndman, stl decomposition before forecasting is&nbsp;&nbsp;beneficial for high frequency time series. I included&nbsp;different values for s.window because this controls how fast the seasonality can change, and this can be important since some departments' seasonalities are pretty stable, while others can vary across years. I also included various options of Box-Cox and optimum criterion in the algorithm. The final prediction comes from the simple average of the two best models.</p>
<p>3.</p>
<p>If you look at the seasonality line graphs that I attached, in some departments such as deaprtment 1, they have a strong Easter effect, so I adjusted them by regression dummies. Also, 2012 Christmas&nbsp;week ended earlier than previous years in terms of which day in December, so in other words, week 152 stole some sales from week 151. I also attach a seasonality heat map of all sales in department 1 where I found beautiful (but not as useful as the line graphs...)</p>
