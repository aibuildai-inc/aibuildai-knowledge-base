# Public 1125 - > Private 48 solution

Competition: godaddy-microbusiness-density-forecasting
Rank: #48
Source: https://www.kaggle.com/c/godaddy-microbusiness-density-forecasting/discussion/417946

Methods that are beneficial for prediction:

Outlier smoothing

Coefficient adjustment + approximation
active: The raw count of microbusinesses in the county The target to be predicted is population density = active / population. The population statistics change every January. The competition requires predicting the population density in 2023, which has changed in the denominator of population density. Based on the existing data, use (2023 predicted value * 2020 population) / 2021 population to estimate the coefficient of population trend change, and then correct the predicted value with this coefficient. Because the numerator part active is originally an integer, the predicted value multiplied by the population is usually a continuous value, such as 1.3, while the actual value may be 1, and rounding can help increase accuracy. The predicted value after coefficient correction should be multiplied by the 2021 population, then rounded, and then divided by the 2021 population for further correction.

Optimal shift time = 12

Create lagged features for the past 12 months, if it is 24 or 6, it will make the effect worse, especially 24.

The state and county fields are label encoded, plus the cfips field (equivalent to state + county), which are beneficial for the prediction results. 
Rate of change effect is better
For predicting n + gap tasks, change the original density continuous value to be predicted to the growth value between the current value and the previous gap months.

Make some lagged features for the original density continuous value, and then add some lagged features for this growth value, which improves the prediction accuracy

After converting the density target value into a growth value, LightGbm and Xgboost models need to change the default objective parameter and use pseudo huber loss as the optimization indicator.


========

对预测有增益的方法：
# 异常值平滑


# 系数调整 + 求近似值
active：The raw count of microbusinesses in the county
要预测的目标为人口密度 = active / 人口。 人口统计值每年一月会发生变化。竞赛要预测2023年的人口密度，在人口密度分母上已经发生了变化。基于已有的数据，使用（2023年预测值 * 2020年人口）/ 2021年人口 估算出人口趋势变化的系数，再将该系数修正预测值。
因为分子部分active本来是整数，预测值乘以人口通常是连续值，譬如1.3，而实际值可能是1，取整后可以帮助精度增加。系数修正后的预测值要乘以2021年人口，然后取整，再除以2021年人口，做进一步修正。

# 最优shift time = 12

创建过去12个月的滞后特征，如果是24或者6，都会让效果变差，尤其是24.
# state和county两个字段做label encoding，加上cfips（等同于state + county）字段，都对预测结果有增益。
# rate of change 效果变好
1.预测n + gap的任务，将原本要预测的密度连续值， 改成当前值与上gap个月的增长值。

2.对原本对密度连续值做一些滞后特征，再加上对该增长值做滞后特征，对预测准度有提升
3.将密度目标值转换成增长值后，LightGbm和Xgboost模型需要更改默认的objective参数，使用pseudo huber loss作为优化指标.
