# Imaginary Rudolph Prize Solution 🦌

Competition: santa-2020
Rank: #1
Source: https://www.kaggle.com/c/santa-2020/discussion/218453

# Overview

- Predict the initial threshold of each bandit with two LGB models at each step.
- The weighted sum of the two predictions is multiplied by decay to get the predicted value of the current threshold.
- Select the bandit with the highest predicted value.

The notebook is [here](https://www.kaggle.com/nagiss/imaginary-rudolph-prize-solution).


# Model

I used the log of 3000 games obtained via the API as training data.

The first one is a simple model that only considers exploitation. I used RMSE as the loss function. The features I used were the following 10:

1. Current step.
2. The number of times the opponent has chosen the bandit two times in a row.
3. The number of different bandits the opponent has chosen so far.
4. The maximum value of 8. for all bandits.
5. Gini coefficient for the bias of the opponent's bandit selection.
6. The number of times my agent have chosen the bandit.
7. The expected value of the threshold, calculated using only the results of my choice.
8. The number of times the opponent has chosen the bandit.
9. The number of times the opponent has chosen the bandit. \\(corrected for 1 / decay at time of choice)
10. The number of steps that have passed since the opponent last chose the bandit.

The second model takes into account both exploration and exploitation. This model also uses RMSE as the loss function, but trains the objective variable as ( 1.02 ^ {threshold} ) and uses ( \log_{1.02} outputvalue ) as the threshold prediction. This is equivalent to using a loss function where the penalty is larger when the objective variable is estimated smaller.

Here is an example of how this transformation of the objective variable works. Consider a bandit that no one has chosen once yet. In this case, the threshold is equal probability from 0 to 100, so in a normal model, the predicted value would be $$ \frac{1}{101} ( 0 + 1 + 2 + \dots + 99 + 100 ) = 50. $$ However, with a transformation of the objective function, the predicted value would be $$ \log_{1.02} \left( \frac{1}{101} (1.02 ^ 0 + 1.02 ^ 1 + 1.02 ^ 2 + \dots + 1.02 ^ {99} + 1.02 ^ {100} ) \right) \approx 58.15. $$

The features used are almost the same as those in the first model, but 7. are calculated with the aforementioned transformation.

At the very beginning of the episode, I mix the predictions of the two models in the ratio -0.2 : 1.2. At each step, the ratio of the first model is increased, and the final ratio is 1 : 0.


# Validation

In order to measure the strength of the new agent, I basically had it play about 1000 games against the strongest agent in my possession. I did not use the average of the expected rewards because I thought that there might be agents who sometimes fail very badly but have a good win rate.

# What did not work

I also considered strategies to avoid giving information to the opponent, but they all failed. I think I should have tried a few more, since the top-ranked teams seem to do this and maintain a high win rate.

# Other strategies

#### Spamming

In this competition, the strength of the agents is important, but luck is also an important factor. I submitted the maximum number of agents to make the most of the luck factor.

#### Information gathering

To learn more about the environment of simulation competitions, I scoured the discussions of past competitions. As a result, I was able to find the following information.

- The existence of an API for retrieving replays
- How to submit a tar.gz
- Overview of the matching algorithm

The following information also seemed to be a little difficult to find.

- The agent log, which shows the standard output and error output of the agent during the game
- You can check the details of the game implementation on github

It would have been nice to have a single page with all this information.

#### OverageTime

By looking at the OverageTime of the replays in the rankings, it was possible to guess if each agent was loading any statistics or models. \\(But it didn't affect my strategy.)

-----------------------------------------------------------------------------

The following is the Japanese version. \\(The contents are the same.)

-----------------------------------------------------------------------------

# 概要

- 毎ステップ、2 つの LGB モデルで各バンディットの初期スレッショルドを予測
- 2 つの予測値の重み付き和に、decay を掛けて、現在のスレッショルドの予測値とする
- 予測値の最も高いバンディットを選ぶ

notebook は[こちら](https://www.kaggle.com/nagiss/imaginary-rudolph-prize-solution)です。


# モデル

APIで取得した 3000 戦の対戦ログを学習データに使いました。

1つ目は、 exploitation のみを考慮するシンプルなモデルです。損失関数には RMSE を使いました。使った特徴量は以下の10個です。

1. 現在のステップ
2. そのバンディットを相手が 2 回連続で選んだ回数
3. 現在までに相手が選んだバンディットの種類数
4. すべてのバンディットに対する 8. の最大値。
5. 相手のバンディット選択の偏りを表すジニ係数
6. こちらがそのバンディットを選んだ回数
7. こちらが選んだときの結果だけを利用して計算したスレッショルドの期待値
8. 相手がそのバンディットを選んだ回数
9. 相手がそのバンディットを選んだ回数（選んだ時点での 1 / decay で補正）
10. 相手がそのバンディットを最後に選んでから経過したステップ数

2つ目のモデルは、exploration と exploitation の両方を考慮します。このモデルでも損失関数として RMSE を使いますが、目的変数を ( 1.02 ^ {threshold} ) として学習を行い、( \log_{1.02} outputvalue ) をスレッショルドの予測値とします。これは、目的変数を小さめに見積もったときにペナルティがより大きくなるような損失関数を使うことと等価です。

この目的変数の変形がどのように動作するかの例を示します。まだ誰も一度も選んでいないバンディットを考えます。このとき、スレッショルドは 0 から 100 まで等しい確率であるので、普通のモデルであれば、予測値は $$ \frac{1}{101} ( 0 + 1 + 2 + \dots + 99 + 100 ) = 50 $$ となることが期待されます。しかし、目的関数の変形を行った場合、予測値は $$ \log_{1.02} \left( \frac{1}{101} (1.02 ^ 0 + 1.02 ^ 1 + 1.02 ^ 2 + \dots + 1.02 ^ {99} + 1.02 ^ {100} ) \right) \approx 58.15 $$ となります。

使った特徴量は1つ目のモデルのものとほとんど同じですが、7. は前述の変形を行って計算しています。

エピソードの最序盤では、2つのモデルの予測値を -0.2 : 1.2 の割合で混ぜます。ステップが進むごとに 1 つ目のモデルの割合を上げ、最終的に 1 : 0 の割合になります。


# 検証方法について

新しいエージェントの実力を測る際、基本的には、自分の所持する最も強いエージェントと 1000 戦程度の対戦をさせました。各対戦の勝敗は expected rewards を使用して判定し、半数以上で勝利すればモデルが改善したと判断しました。expected rewards の平均を使わなかったのは、時々物凄く失敗するが勝率は良いようなエージェントが存在するかもしれないと考えたからです。

# うまくいかなかったこと

相手に情報を与えないための戦略も考えましたが、全て失敗しました。最上位勢はこれをやって高い勝率を維持しているようなので、もう少し試せばよかったと思っています。

# その他の戦略

#### 投稿エブリデイ

このコンペはエージェントの強さも重要ですが、やはり運も重要な要素です。運の要素を最大限利用するため、最大数のエージェントを投稿しました。

#### 情報収集

シミュレーションコンペの環境について知るために、過去のコンペの Discussion を漁りました。結果として、

- リプレイを取得するための API の存在
- tar.gz を提出する方法
- マッチングアルゴリズムの概要

といった情報を知ることができました。また、

- 対戦中のエージェントの標準出力・エラー出力を agent log を見ることができる
- github でゲームの実装の細部を確認できる

といった情報も、知ることが少し難しかったように思います。このあたりの情報はひとつにまとまっているページがあると良かったと思います。

#### OverageTime

ランキング上でリプレイの OverageTime を見ることで、各エージェントが何らかの統計データやモデルを読み込んでいるか推測することが可能でした。（しかし、私の戦略に影響を与えることはありませんでした。）
