# Thank You and #2 rank model

Competition: walmart-recruiting-store-sales-forecasting
Rank: #2
Source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/discussion/8023

<p>Congrats David and all the contestants.</p>
<p>Personally, I&nbsp;would like to personally take this opportunity to &nbsp;thank &nbsp;Kaggle community, It has been an enjoyable experience. I have learnt a lot from everyone in this website.</p>
<p>With regards to my model, I'll upload a detailed explanation on &quot;how and why&quot; of my approach.&nbsp;I used a hybrid approach of statistical and machine learning methods.</p>
<p>I used SAS (for data prep/ARIMA/UCM) and R (for the remainder models) together. I used weighted average and trimmed mean of following &nbsp;6 methods. The goal from &nbsp;the beginning was to build a robust model that will be able to withstand uncertainty.</p>
<p>Statistical Methods:</p>
<p>1.&nbsp;Auto-regressive Integrated Moving Average (ARIMA)</p>
<p>2. Unobserved Components Model (UCM)</p>
<p>Machine Learning Methods:</p>
<p>3. Random Forest</p>
<p>4. Linear Regression</p>
<p>5. K nearest regression</p>
<p>6. Principle Component Regression</p>
<p>My model did not use any features. I simply used past values to predict future values.&nbsp;</p>
<p>With Regards to variables (features) I used week of the year (1 thru 52), this would capture almost all the lag and lead effects of holidays except for new year which was moving and one other holiday. I built individual models for each department. I weighted holidays for stores with high growth rate vs. prev year differently than the stores without high growth.</p>

<p>In the next week or so I'll try to upload&nbsp;detailed explanation.</p>
