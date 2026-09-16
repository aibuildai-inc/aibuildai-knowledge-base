# Solution Sharing

Competition: the-winton-stock-market-challenge
Rank: #2
Source: https://www.kaggle.com/c/the-winton-stock-market-challenge/discussion/18584#116236

Hi chenhan zhang,

I hope you are fine.

It was not a data leak. It is important to say. It was about an important and public information in the dataset that we could use it without to know about returns. People from Winton explained a little about the main feature. It was the Feature 7. The mistake was in the data split and they changed the dataset for this reason. After this, everything was ok.

In my opinion, Feature 7 represented a mapping between days and sectors. For example, 03/04/2012 x Gas and Oil -> 678965 F7 value. It is only an example. It does not represent a right value.

I found it using my excel and a person correlation between stocks (using intraday series). With this information, you can get many points if you understand about stock exchanges.

Many people knew that Feature 7 was important. However, they do not know why and how to use it. I saw some people in the forum recommending to drop F7. It was a big mistake. 

Other people created wrong groups to validate their models. You cannot use a "future" information to validate your training. And F7 could help you to spilt data in the right way. For this reason, people were talking about very good CV results but bad results in the public LB. Maybe the majority saw it in the home tests. I saw it in the first days. And I realized that something was wrong and I started a process to find the "secret".

There were many ways to work with this information. I tried to assume high risks to get high returns. The best that I find was change few points in D+1. The equation to evaluate quality of solution was not good, in my opinion, and I explored it as well. I did thousands of tests to explore this kind of "mistake".

A wrong equation + a good information in the dataset (F7) made me to change only 74 points.

It was the "secret". Many people said bullshits about my strategy in the forum. Anyway, they have failed. I only studied a lot the dataset and the function.

There were many ways to work with F7 information. I only chose one of them. My strategy was to predict volatility, controlling risks. It was not about returns, that is impossible, in my opinion.

Best regards...
