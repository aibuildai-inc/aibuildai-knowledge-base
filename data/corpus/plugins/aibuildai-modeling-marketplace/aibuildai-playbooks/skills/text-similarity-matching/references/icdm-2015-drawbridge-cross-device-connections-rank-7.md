# 1st place solution summary

Competition: icdm-2015-drawbridge-cross-device-connections
Rank: #7
Source: https://www.kaggle.com/c/icdm-2015-drawbridge-cross-device-connections/discussion/16122#90342

Thanks for sharing....
 Another good feature for reducing the device-cookie ip-pairs is: Device Counts on IP and Cookie Counts on IP, just count how many devices and cookies you will see on each IP, the less you see the more the probability they share handles. Just see how powerful this feature is: if you put a threshold like 3 for devices and 40 for cookies, you will  filter out 85% of the ip pairs table and keeping 98% of same handles. Also another good feature is for each cookie if you divide the Cookie_freq_on_IP by the maximum frequency of that cookie in the whole data and call it RATIO_FREQ_COOKIE, not surprisingly this variable will have value "1" in 85% of the good pairs and in 95% of the good ones with value greater than "0.02" and it make sense! because this cookie appears in may places but the most frequent ones appears in the proper device pair. By doing so you will get 77% accuracy on LB "without" any learning algorithm.
