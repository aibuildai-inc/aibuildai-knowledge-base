# 3rd Place Solution (convert the COVID affected 2020 week8-week32's emission)

Competition: playground-series-s3e20
Rank: #3
Source: https://www.kaggle.com/c/playground-series-s3e20/discussion/433822

Thanks to the organizers and participants of this competition!
# 2020 week 8-32 emissions are outliers
Inspired by [ambrosm's post](https://www.kaggle.com/code/ambrosm/pss3e20-eda-which-makes-sense#What-the-lockdown-implies-for-cross-validation), the Year-Over-Year Growth by Quarter show's a COVID affected period from 2020 Q2,Q3,Q4 and 2021 Q1.
I made a more detailed Year-Over-Year Growth by Month, as we can see that the most affected months are 2020 Mar, Apr, May, Jun, Jul, Aug with a double digits decrease in emission. That means the corresponding weeks 8-32 in 2020 are outliers! And from 2020 Sep, the decrease trend goes into digits, so it can be assumed that the affect of COVID isn't high compared to previous months, this can be regard as the recovery.

So, based on this, split the training dataset into two:

# Convert outliers to the range of Non-COVID affected period
Inspired by [KACPER RABCZEWSKI's post](https://www.kaggle.com/code/kacperrabczewski/rwanda-co2-step-by-step-guide), this post's idea is transfer the 2020's emission to the average level of 2019 and 2021.
I did the conversion by following steps:
- Calculate the average emission by week in the Non-COVID dataset.
- Calculate the average emission by week in the COVID dataset.
- Calculate the ratio.
- Convert the COVID dataset’s emission by multiplying the ratio.
- Replace the training dataset’s emission value by the new value of the COVID period.


# Feature selection, training samples selection
Select week_no <= 48 samples for training, which is the same week range as the test 2022.
Select 3 features ['latitude', 'longitude', 'week_no'] which has no missing values.


# Ensemble Model


# Submission fix
[AMBROSM's discussion topic](https://www.kaggle.com/competitions/playground-series-s3e20/discussion/428791), I multiply the result by the constant 1.07.
[CHUN FU's notebook](https://www.kaggle.com/code/patrick0302/find-and-fix-the-error-bug), fix the bug at ['longitude'] == 29.321.
