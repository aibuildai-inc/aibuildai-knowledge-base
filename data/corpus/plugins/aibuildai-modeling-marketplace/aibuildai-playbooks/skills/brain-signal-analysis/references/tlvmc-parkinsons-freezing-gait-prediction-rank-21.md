# 21st place solution: Conv1d with denoising

Competition: tlvmc-parkinsons-freezing-gait-prediction
Rank: #21
Source: https://www.kaggle.com/c/tlvmc-parkinsons-freezing-gait-prediction/discussion/415975

My 21st solution is based on a great public notebook [here](https://www.kaggle.com/code/coderrkj/parkinson-fog-pred-conv1d-separate-tf-model). The chosen one of my submissions is an ensemble of 5 similar models(**public LB: 0.433, private LB: 0.324**). My best private LB is a single model(**public LB: 0.421, private LB: 0.327**), I publish it [here](https://www.kaggle.com/code/takanashihumbert/gait-single-models-inference/notebook).
## model
The basic structure of my model composes of 3 different `Conv1d` blocks, then `concatenate` and `flatten`, finally with a 4 labels output.

The features are **Time_frac**, **AccV**, **AccML**, **AccAP**, **V_ML**, **V_AP**, **ML_AP**(the last 3 means the difference value between two of AccV, AccML, and AccAP). And I know **Time_frac** is a powerful(improve nearly 0.1) but controversial one. I use multi-class, so the labels are **StartHesitation**, **Turn**, **Walking**, and **Normal**.
## denoising
My denoising code is very simple:
```
def wavelet_denoising_2(x, wavelet='db4'):
    coeffs = pywt.wavedec(x, wavelet, mode="per")
    coeffs[len(coeffs)-1] *= 0
    coeffs[len(coeffs)-2] *= 0
    result = pywt.waverec(coeffs, wavelet, mode='per')
    if len(x)%2==1:
        result = result[:-1]
    return result
```
## window size
```
defog_window_size = 200
defog_window_future = 50
tdcsfog_window_size = 256
tdcsfog_window_future = 64
```
I think that's all, nothing special. Thanks. :)
Looking forward to top and interesting solutions！
