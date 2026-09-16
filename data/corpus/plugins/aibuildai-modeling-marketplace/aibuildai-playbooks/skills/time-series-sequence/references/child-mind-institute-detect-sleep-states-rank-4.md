# 4th Place Solution - Nikhil's Part (Modified Unet + Transformer and Weighted Box Fusion)

Competition: child-mind-institute-detect-sleep-states
Rank: #4
Source: https://www.kaggle.com/c/child-mind-institute-detect-sleep-states/discussion/459637

Firstly thank you to my teammate @ryotayoshinobu for the amazing performance. Please read his part of the [solution](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/discussion/459597).
I am really glad I teamed up with him and learnt a lot

A big thanks to the organizers as well for such an interesting competition. A really tough one.

You can find the solution code here:

https://github.com/nikhilmishradevelop/kaggle-child-mind-institute-detect-sleep-states

# Solution Summary

**Validation Strategy**: GroupKFold on Series Ids

**Model Inputs**:  17280 x n_features length sequences as input (17280 = 12 steps_per_minute x 60_minutes * 24 hours)
**Model Outputs**: 17280 x 2 (one for onset and other for wakeup)
**Model Type**: Regression Model
**Loss Type**: Cubic Loss i.e. abs(y_true-y_pred)**3

The last remaining part of the sequences with length < 17280 were padded to make them equal to 17280

My solution is a modified UNET (averaged over 4 models, 2 LSTMs and 2 GRUs) using normalized Gaussian targets similar to what @tolgadincer described. A big thanks to him for sharing a good method, early on in this competition.

## Features:

1. Original Sequence Features: Enmo, Anglez
2. TimeStamp Features: Hour and Weekday
3. Derived Sequence Features: Anglez difference, Enmo Difference, HDCZA features etc)

Good Features help in faster convergence and much better scores, so adding good features were important (Read penguin's solution for some other good features)

**My final 4 model ensemble results**: 

**CV**: 0.828
**Public LB**: 0.789
**Private LB**: 0.841


# **Model Architecture**


[Model Architecture].png?generation=1701841137052567&alt=media)


##Patching to reduce sequence length

Since 17280 is a very long sequence length (makes training very slow and harder to train), we reduce the sequence length by patching it.

**Input:**  17280 * n_features (34)
**Patch Size:** (Used different patch sizes 3, 4, 5, or 6 in different models).

**Modified Sequence Length:** 17280 to 17280//Patch_Size sequence length

**Modified Feature Size:** k * 4 * num_features (where k is a Dense layer output dim)

## Modified Unet Part

***UNET Encoder -> Bottleneck -> Transformer -> GRU or LSTM -> Unet Decoder***

Each layer of UNET encoder had concatenated connections to the 1st layer

**Initial Output size:** 17280//Patch_Size, 2 * Patch_size

**Reshaped Output Size:** (17280, 2)




# Post Processing

Initially I was doing just doing simple peak detection when I was competing solo.

I wanted to, but could not formulate a good way to rerank model predictions using lightgbm, but @kmat2019 is a great read for that.

Thanks to @ryotayoshinobu , I started applying NMS, and finally experimented with a kind of  WBF algorithm and made it work. This WBF algorithm gave a score of 0.79 to 0.793 on last day.

But it was worse by 0.001 in our private LB lol, so WBF hurt.

WBF Working

1. **Initialization and Convolution**: The function starts by applying convolution with the specified `convolution_kernel` to smooth the data.

2. **Peak Detection Loop**: It iteratively searches for peaks in the data. The loop runs until the maximum count (`max_count`) is reached or the peak value falls below a threshold (`max_thresh`).

3. **Adaptive Window and Weight Calculation**:
   - A dynamic window size around each peak is determined based on the current maximum value's power (`curr_max_power`) and a distance parameter (`k_dist`).
   - Depending on the `section_weight_method` (logarithmic or linear), weights are calculated for each section of the data around the peak.

4. **Weighted Average and Score Calculation**: For each detected peak, a weighted average is calculated to determine the score of the peak. 
This is done by considering the peak and k of its neighbours. This score is influenced by the method of weight calculation and other hyperparameters like `log_base`, `log_scale`, and `weight_coeff`.

5. **Suppression and Updating Predictions**:
   - After each peak is detected, the function suppresses the neighboring values to avoid detecting the same peak multiple times. This is controlled by `overlap_coeff` and `preds_reduction_power`.
   - The indices and scores of detected peaks are stored and returned after the loop completes.



I will share the implementation of the WBF part code, you can see if tuning these hyperparameters help in your models too. This accepts predictions per series id.

(I found hyperparameters by a mix of manual and automated hyperparamter tuning approach)

```
def wbf_nikhil(preds_orig, max_thresh=0.1, max_count=700, hyperparams=None):
    k_dist = hyperparams['k_dist']
    log_base = hyperparams['log_base']
    log_scale = hyperparams['log_scale']
    curr_max_power = hyperparams['curr_max_power']
    weight_coeff = hyperparams['weight_coeff']
    convolution_kernel = hyperparams['convolution_kernel']
    section_weight_method = hyperparams['section_weight_method']
    preds_reduction_power = hyperparams['preds_reduction_power']
    overlap_coeff = hyperparams['overlap_coeff']
    min_distance = hyperparams['min_distance']
    
    preds = preds_orig.copy()
    preds = np.convolve(preds, convolution_kernel, mode='same')

    count = 0
    indices = []
    scores = []

    while count < max_count:
        curr_max_idx = np.argmax(preds)
        curr_max = preds[curr_max_idx]

        if curr_max < max_thresh:
            break

        k = int(k_dist - max(min_distance, (curr_max**curr_max_power)))

        start_idx = max(curr_max_idx - k, 0)
        end_idx = min(curr_max_idx + k + 1, len(preds))

        section = preds[start_idx:end_idx]

        # Different weight calculation methods
        distances = np.abs(np.arange(len(section)) - k)
        if section_weight_method == 'logarithmic':
            weights = 1 / (log_base ** (distances / (k * log_scale)))
        elif section_weight_method == 'linear':
            weights = 1 - (distances / k) * weight_coeff
        # Add more methods as needed

        weighted_avg = np.sum(section * weights) / np.sum(weights)

        scores.append(weighted_avg)
        indices.append(curr_max_idx)

        preds[start_idx:end_idx] *= ((1 - weights * overlap_coeff))**preds_reduction_power

        count += 1

    return indices, scores
```

## Final Ensemble


Final ensemble was a blended weight of my and Penguin's subs regression   postprocessed using WBF

Final_Sub = WBF(Penguins_Predictions * 0.25 + Nikhil's Predictions*0.75)

Since our predictions had slightly different scales of regression targets, Penguin's Predictions were first scaled by using a simple power transform i.e Penguin's_Predictions ** 0.7

**Ensemble CV:** 0.835
**Ensemble Public LB:** 0.793
**Ensemble Private LB:** 0.845


P.S: This competition is the first time I did not use gradient boosting, even after I knew it worked so well, me being a tabular guy. I mistake I guess. I love lightgbm and will definitely use one in the next competition.

Congratulations to all the the top performers, it was a good fight, till it lasted :)
