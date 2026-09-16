# Solution(public:0.471,private:0.505 )

Competition: recruit-restaurant-visitor-forecasting
Rank: #1
Source: https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/discussion/49129

**plantsgo approach summary:**

**Models**

I have three same models with the same features:

model_1(step=14): use 14 days for target and slip 30 times,so I have 14*30 samples.

model_2(step=28): use 28 days for target and slip 15 times,so I have 28*15 samples.

model_3(step=42): use 42 days for target and slip 10 times,so I have 42*10 samples.

The reason is I don't want my model only focus on the days before or the days last.

**Features:**

1.visit_info:(21,35,63,140,280,350,420) days before groupby：

   air_store_id,weekday,holiday,air_area_name,air_genre_name like:

   (air_store_id,weekday),(air_store_id,holiday),(air_area_name,air_genre_name ,holiday) and so on.

2.reserve info:(35,63,140) days before groupby：

air_store_id,weekday,holiday,air_area_name,air_genre_name

**Ensembel:**

Use (xgb,lgb,nn) 0.7*lgb+0.2*xgb+0.1*nn: only improved 0.0002 offline.

0.334*model_1+0.333*model_2+0.333*model_3:improved 0.002 offline.

**Code:**

https://www.kaggle.com/plantsgo/solution-public-0-471-private-0-505
