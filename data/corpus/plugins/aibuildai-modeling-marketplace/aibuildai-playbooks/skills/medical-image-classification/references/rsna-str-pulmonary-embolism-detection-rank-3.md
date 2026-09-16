# 3rd place 0.156 public lb, 0.148 private lb

Competition: rsna-str-pulmonary-embolism-detection
Rank: #3
Source: https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/193424

Edit: Github added.

First of all, congrats to all the winners.

Our solution is quite straight forward. We mostly use the same tricks from last year's RSNA-IHD with some updated modules for performance improvements, label consistency, and exam-level prediction.

The solution can be divided into 3 main parts:

Part one is single-image training. We used CNN to predict `pe_present_on_image` probability for each slice and modify the CNN model by adding a FC layer with 7 units before the final binary prediction layer. This layer act as a embeddings-generator layer and the final binary will just be a linear function of these 7 inputs. We chose 7 because they actually are fine-grained labels of `pe_present_on_image` which are combinations of slice-level labels and exam-level labels (`pe_present_on_image` and `rv_lv_ratio_gte_1` or `pe_present_on_image` and `central pe`  for example). We also tried to grain the label finer but no luck.

Part two is sequential slice-level models. From each slice in part one, we get a 7-d embeddings vector, concatenating 31 consecutive slice's embeddings vector and we have a 31-by-7 image as embeddings feature map of the center slice (in other words, for each slice, we also look at 15 slices before it and 15 slices after it). For edge cases, padding is used. Part two model is a module combining a simple shallow CNN with no pooling and a sequential model with two bi-directional LSTM layers. We get the output of this model as the final prediction of each slice, and with reversing augmentation and two models (CNN and LSTM), we have 4 outputs. Concatenate them all and we have a final 32-d embeddings vector for each slice.

Part three is the exam-level CNN models. We just stacking the 32-d embeddings vector of all the slice in an exam and chose 1024-by-32 as the common image size. For exams with less than 1024 slices, we zero padding and for exams with more than 1024 slices, we truncate it. This model will predict 9 exam-level labels.

My training source code:
https://github.com/moewiee/RSNA2020-Team-VinBDI-MedicalImaging

You guys can also take a look at my inference kernel:
https://www.kaggle.com/moewie94/rsna-2020-inference
