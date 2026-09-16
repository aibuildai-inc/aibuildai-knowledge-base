# First Place Entry

Competition: walmart-recruiting-sales-in-stormy-weather
Rank: #1
Source: https://www.kaggle.com/c/walmart-recruiting-sales-in-stormy-weather/discussion/14452

<p>Thank you all people around this competition, I'm a newbie in data science and it was the first challenge for a non-playground competition, so I'm really surprised and glad to win.</p>
<p>I'm not great at English, so wrote this method description in itemized style.</p>
<p><strong>Train model</strong></p>
<p>1. Exclude item/stores whose units are all zeros.</p>
<p>2. For each item/stores,<br> apply curve fitting by R ppr function (projection pursuit regression).<br> y = log1p_units, x = days from 2012-01-01</p>
<p>here, data on 2013-12-25 are excluded. (because units are almost all zeros)</p>
<p>3. Train linear model with lasso using vowpal wabbit.<br> y = log1p_units - ppr_fitted</p>
<p>features :<br> - A : weekday, is_weekend, is_holiday, is_holiday_and_weekday, is_holiday_and_weekend<br> - B : item_nbr<br> - C : store_nbr<br> - D : date<br> - E : year, month, day<br> - F : is_BlackFriday-3days, -2days, -1day, is_BlackFriday, +1day, +2days, +3days<br> - G : weather features (is preciptotal &gt; 0.2, depart &gt; 8, depart &lt; -8)<br> - interactions A*B A*C B*E C*E B*F C*F</p>
<p>here, below are excluded:<br> - on 2013-12-25<br> - moving average(21 elements, centered) is zero.</p>
<p>4. Mark dates as &quot;too much zeros&quot; where both sides are many successive zeros.<br>4-1. for dates whose units are not zero, calculate minimum of both side successive zeros (= min_side_zeros).<br>4-2. for each item/stores, <br> calculate maximum of min_side_zeros (= max_min_side_zeros), floor and ceiling by 1 and 9.<br>4-3. for each item/stores,<br> mark dates as &quot;too much zeros&quot; where both sides are successive zeros more than max_min_side_zeros.</p>
<p><strong>Prediction on test set</strong></p>
<p>predicted_log1p = ppr_fitted(<em><strong>train-2</strong>) +</em> linear model predicted(<em><strong>train-3</strong></em>)<br>predicted = exp(predicted_log1p) - 1</p>
<p>here, below are predicted as zero.<br>- item/stores whose units are all zeros. <br>- on 2013-12-25<br>- moving average(21 elements, centered) is zero.<br>- &quot;too much zeros&quot; (<em><strong>train-4</strong></em>)</p>
<p><strong>Comments</strong></p>
<p>The core idea is very simple like that:<br>1. Create a baseline for each item/stores.<br>2. Apply linear regression using vowpal wabbit with many features.</p>
<p>As for baseline:<br>- R ppr functions fit really nice on almost all item/stores (can be improved on some item/stores). <br>- At first I used moving average. It worked, but fluctulates too much or catch too distant value.</p>
<p>As for features:<br>- weekday is the most important<br>- month periodicity is on some store/items<br>- around Black Friday sales fluctuates a lot<br>- weather features are not effective almost at all<br>&nbsp;&nbsp; In the data, people go shopping as usual however much it rains. <br>&nbsp;&nbsp; It's not natural, so I guess weather data came from different stations.</p>
<p>Considering successive zeros was my final push, it slightly improved the score.</p>
<p><strong>Codes</strong></p>
<p>uploaded on github, <a href="https://github.com/threecourse/kaggle-walmart-recruiting-sales-in-stormy-weather">https://github.com/threecourse/kaggle-walmart-recruiting-sales-in-stormy-weather</a></p>
