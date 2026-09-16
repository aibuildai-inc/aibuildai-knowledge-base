# [17th] 🔥 PERFECT cv 🔥  & improve step by step [中英]

Competition: indoor-location-navigation
Rank: #17
Source: https://www.kaggle.com/c/indoor-location-navigation/discussion/240077

Hello everyone,


This competition is very good, thanks to XYZ10 Technology, thanks to the organizer of kaggle, I love kaggle so much. Completing this competition I think is a very good growth for me. I have learned (reviewed) a lot of things, including: tensorfloor, pytorch, multi-threading skills, code specifications, standardized experimental records, and the importance of cv . I haven't made any breakthroughs in algorithms and features. The only thing I haven't seen anyone discuss is the construction method of my cv. I will share it later.
# step by step
Now let’s take a look at my main improvement points in the past month:

| Mainline | public |
| --- | --- |
Lstm public kernel is used with simple parameter adjustment, also use cost min+snap2grid | 4.72
A fake test is constructed by simulating the characteristics of test data|
Interpolate the wifi timestamp of the training set, the test set is still the timestamp of the waypoint|4.45
Drop out wifi log with an interval greater than 5 seconds | 4.20
By find a timestamp very close to the train, fix start and end point| 4.18
The training set is unchanged, the test set is also the wifi timestamp, and then de-interpolated to get the waypoints | 4.16
Do cost min at the wifi timestamp level, interpolate to get wp, <br>snap, fix the start and end points, and finally do cost+snap again| 4.0x-390
4 fold use fake test strategy| 3.82
More careful fix start and end points | 3.67
Duild delta modeling of wifi timestamp granularity improves the result of wifi cost min | 3.46
Duild delta modeling of waypoint timestamp granularity improves the result of wp cost min| 3.36
Do cost min at the wifi timestamp level, interpolate to get wp, <br>snap, fix the start and end points, and twice cost+snap+fix| 3.23
Do cost min at the wifi timestamp level, interpolate to get wp, <br>snap, fix the start and end points, and three times cost+snap+fix | 3.19

# fake test
When I did simple path group Nfold for the first time, there was no improvement. I realized that the distribution of the training set and the test set was inconsistent. I conducted a series of probes and found that the test set has some characteristics that should be artificially limited. **The txt size of the test set>=2M (look in linux), the time span>=60s, and the number of path points>=5**. In the first half of the month, i also selected 626 paths as fake tests using above restrictions . No matter whether post-processing is performed or not, my fake test scores are consistent with the lb scores. In the last week, I used 4-fold cross-validation. I also made the above restrictions to make my cv more consistent. Later, I checked the score of private, which also conformed to the same trend.


# Failed attempt
1. Directly use png image features as end-to-end input
2. Use pytorch to reproduce the network structure and lb score
3. Use MLP to replace lstm that is not a time series
4. Use all site data
5. Try sequential RNN, but it does not work well and it is difficult to converge 
. . . . . .

# I have a question
I haven't figured out why the lstm model with a sequence length of 1 works so well, and why can't mlp replace it. . .
**If you have any comments on this. I will be very grateful.**

<hr>
<hr>

大家好，

这个比赛非常棒，感谢十域科技，感谢kaggle主办方，我太爱kaggle了。完成这个比赛我觉得是对我个人的一次非常好的成长，我学到了（复习了）非常多的东西，包括：tensorfloor、pytorch、多线程技巧、代码规范、规范的实验记录、cv的重要性。我没有在算法和特征上有什么突破，唯一有一点我没有看到有人讨论的是我cv的构建方法。后面我将会分享到它。

# step by step
现在先看一下我这一个月来的主要提升点：

| 主线| public | 
| --- | --- |
使用了 lstm public kernel，进行简单调参，使用cost min+snap2grid|  4.72
模拟test data 的特点构造了一个 fake test|
对训练集wifi时间戳进行插值，测试集还是路径点的时间戳|4.45
过滤掉间隔大于5秒的wifi | 4.20
寻找时间非常接近的点，修复开始结束点  | 4.18
训练集不变，测试集也是wifi时间戳，再反插值得到路径点|  4.16
做wifi时间戳级别的cost min，插值得到wp，<br>snap，修复开始结束点，最后再嵌套一次cost+snap| 4.0x-390
引入4折 fake test |  3.82
更小心的修复开始和结束点 |   3.67
对wifi时间戳粒度的delta建模提高wifi cost min的结果|   3.46
对wp时间戳粒度误差建模提高cost min|  3.36
做wifi时间戳级别的cost min，插值得到wp，<br>snap，修复开始结束点，嵌套两次cost+snap+fix|   3.23
做wifi时间戳级别的cost min，插值得到wp，<br>snap，修复开始结束点，嵌套三次cost+snap+fix |   3.19

# fake test
第一次做Nfold的时候没有提升，我意识到训练集和测试集的分布不一致，我进行了一系列的探测，发现测试集有一些特点应该是人为限定的。测试集的txt大小>=2M（linux下看），时间跨度>=60s，路径点个数>=5。在前半个月的时候，这个限制同样抽出了626个路径作为fake test，无论是否进行后处理，我的fake test得分和线上的得分都保持一致。在最后1周我使用4折交叉验证，作为valid的那一份数据我同样做了上述限制，使得我的cv更加一致。后面我检查了private的分数情况，也符合相同的趋势。

# 失败的尝试
1.直接使用png图像特征作为端到端的输入
2.使用pytorch复现网络结构和lb分数
3.使用MLP替换并不是时间序列的lstm
4.使用所有site的数据
5.尝试序列RNN，但是它效果不好且难以收敛
。。。。。。

# 我有一个问题
我这一个月始终都没有想明白，为什么序列长度为1的lstm模型工作的这么好，mlp为什么无法替代它。。。
如果您对此发表任何看法。我将非常感激。
