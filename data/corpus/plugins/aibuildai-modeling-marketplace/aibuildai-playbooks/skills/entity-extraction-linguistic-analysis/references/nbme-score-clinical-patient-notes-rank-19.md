# 19th Solution: Pseudo label and PP

Competition: nbme-score-clinical-patient-notes
Rank: #19
Source: https://www.kaggle.com/c/nbme-score-clinical-patient-notes/discussion/322993

I'd like to thank host and kaggle for organizing this great competition! 
And thanks to all participants for the useful codes and interesting discussions!
And also thank you to all of my excellent teammates(@naoism, @yosukeyama, @mst8823)!
<!-- ホストの皆さん主催ありがとうございました．データの処理，エラー分析の重要性を改めて知ることができました．
参加の皆さん有用なコードや興味深い議論ありがとうございました．おかげでより刺激的なコンペになりました．
そして優秀なチームメイトの皆さん，ありがとうございました．様々な議論やアイデアが勉強になりました． -->

## Summary
ensemble of 5 models. some post processing.
cv:0.8971, private lb:0.892
<!-- 5fold, 5モデルのアンサンブル．いくつかの後処理． -->

## MLM pretraining
The accuracy was increased by about 0.002.
- mlm_probability=0.15. using all unlabeled data.
<!-- 基本的に用いることで0.002ほど精度があがりました． -->
<!-- mlm_probability=0.15. すべてのラベルなしデータ． -->

## Models
three deberta-v3-large. one deberta-v2-xlarge. one deberta-v1-large.
The loss function for all models is BCEwithLogitsLoss (we tried jaccard and KLD, but bce had the best cv).
<!-- すべてのモデルの損失関数はBCEwithLogitsLossです(jaccardなども試したがcvはこれがbestであった)． -->

- model1: 
	- deberta-v3-large
	- cv:0.8875, private lb: 0.889
	- including /n charactors like this [discussion](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)

- model2: 
	- deberta-v3-large
	- cv:0.8957, private lb: 0.890
	- including '/n' charactors
	- awp
	- mask augmetation: prob=0.8, ratio=0.03
	- pseudo label
		- Pretraining with pseudo-labels generated from unlabeled data. Finetune with pseudo-labels generated from labeled data.
		<!-- - ラベルなしデータから生成した疑似ラベルでの事前学習．ラベルありデータから生成した疑似ラベルでのfinetune. -->

- model3:
	- deberta-v3-large.
	- cv: 0.890. private lb: - 
	- awp
	- mask augmetation: prob=1.0, ratio=0.2
	- multi sample dropout
	- pseudo label(soft pseudo label)
		- Trained with 11,000 random samples from unlabeled data while maintaining labeled data.
		<!-- - ラベルありデータはそのまま，ラベル無しデータから11000件ランダムサンプルして学習． -->

- model4:
	- deberta v2 xlarge
	- cv: 0.882 private lb: -
	- no mlm.
	- Using bnb for increase the batch size, because small batch size made it difficult to converge.
	<!-- - バッチサイズが小さいと収束しにくいので, バッチサイズを大きくするためにbnbを使用した. -->

- model5:
	- debera v1 large
	- cv: 0.8934, private lb: 0.891
	- multi sample dropout
	- pseudo label
		- Pretraining with pseudo-labels and labeled(train) data. Finetune with labeled data.
		<!-- - 疑似ラベルとラベルありデータで学習．ラベルありデータのみでfinetune. -->

## pseudo label
@naoism implemented it. Thank you very much.
Only model3 uses the inference results from one model.
Other models were using the ensemble results of three models with good cv without pseudo.
<!-- naoismさんが実装してくれました．ありがとうございます． -->
<!-- model3のみ1モデルでの推論結果を利用． -->
<!-- 他モデルはpseudo無しで精度の良かった3モデル，合計３モデルのアンサンブル結果を利用． -->

## ensamble
- Char probabilities level ensemble．Ensemble the ensemble results with 5folds for each model.
- threshold uses the best of cv
- Combining with and without pseudo labels improves accuracy.
<!-- 各モデルごとのfoldでアンサンブルした結果をアンサンブル． -->
<!-- thresholdはcvのベストを利用． -->
<!-- pseudoラベル有りなしを組み合わせることで精度がよくなりました． -->

## post process
- Inference is misplaced if word is not immediately preceded by a space (e.g., "hello) that happen due to offset mapping issues. In such a situation, we fix the beginning of the inference is shifted by one. 
- Similarly, inferences starting with 1 are modified to 0.
<!-- - offset mappingの問題で，単語の直前が空白でない場合(e.g "hello)に推論がずれてしまう問題があったためそれを修正． -->
<!-- - 同様に1から始まる推論を０に修正． -->
- Correction for continuous whitespace
<!-- - 空白が連続する場合修正． -->
<!-- - 知識ベースのpp -->
<!-- - yyamaさんが多くの間違いを探してくれました．ありがとうございます． -->
<!-- - cvは伸びましたがtestデータでの伸びは小さかったです．testにはこのような間違いが少なかったのでしょう． -->
- Knowledge-based pp
	- @yosukeyama found many erros. Thank you very much.
	- e.g 

	```python
	# feature_num 105: Incorrect because it is urine.
	# No-bloody-bowel-movements 
	test.loc[(test["str_annot"].str.contains("urine")) & (test["feature_num"] == 105), "location"] = ""
	test.loc[(test["str_annot"].str.contains("URINE")) & (test["feature_num"] == 105), "location"] = ""
	```
	- cv improved, but the improvement in the test data was smaller, probably because there were fewer errors like that in the test data.

## Others
- The unable to predict "[]" was hurt accuracy, since micro-averaged F1 is the evaluation metric. So We tried multitask learning, but it was difficult to solve the problem because most of the errors were caused by annotation errors.
<!-- - micro-averaged F1が評価指標であるため"[]"を予測できない場合精度へかなり悪影響でした．そこでマルチタスクでの推論を試みましたが，ほとんどがアノテーションミスが起因であったため解決が困難でした． -->
- For the purpose of automatic pp, we built a model that predicted whether the input inference result would have the correct label, but it did not work.
<!-- - 自動ppとしてpn_history、feature_textに加えて、予測されたannotationを入力に正解ラベルがあるかどうかを予測するモデルを作成しましたが，うまく学習できませんでした． -->
- stacking. cv increased by about 0.01, but No effect in test data. 
<!-- - stacking．cvは0.01ほど伸びましたが，lbでは効果がみられませんでした． -->
- The dynamic padding implemented by @mst8823 has accelerated the inference speed, that allowing for ensembles of 25 models or more. Thank you very much.
<!-- - masatoさんが実装したdynamic paddingにより推論速度が早まり，25モデル以上のアンサンブルが可能になりました．ありがとうございます． -->
