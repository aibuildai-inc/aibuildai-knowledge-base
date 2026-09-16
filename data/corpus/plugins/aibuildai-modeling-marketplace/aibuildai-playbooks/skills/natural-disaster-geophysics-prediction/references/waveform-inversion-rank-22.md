# Share some findings

Competition: waveform-inversion
Rank: #22
Source: https://www.kaggle.com/c/waveform-inversion/discussion/587417

Thanks to the organizers for hosting such an interesting and stable competition, and thanks to all the open source developers for their initial contributions.
Considering that my score was not very good, I will not be publishing a solution. However, I would like to share some simple and interesting findings I discovered during the competition, and I hope they will be helpful to you.

## Data generation

The formula for data generation in the OpenFWI paper is incorrect; the correct formula should be:

$$c_{i}(x, y) = c_{i-1}(x, a_{i}sin(2πk_{i}(x+Φ)) + y) \tag{2}$$
$$c_{i}(x, y) = c_0(x+s_i,y+s_i') \tag{3}$$

And the range of parameter selection is not completely random. Specifically, the initial flat velocity map consists of 3 to 8 regions, where the probability of having 3 or 8 regions is half that of having 4 to 7 regions. The curved wave is composed of a mixture of sine waves with periods of 70/1, 70/2, ... 70/5, with the phase randomly sampled from -π to π. I was not able to successfully reverse-engineer the amplitude. 

The paper states that the weight of the background incremental velocity map in the style data generation process is 0.7–0.9, but in reality, it should be 0.1–0.3.

## Large-scale pre-training

With the data generation method, it means that unlimited data can be generated at low cost. I tried pre-training on data that is 100 times the official data, which can increase the cv by about 4. We can expect that more data can achieve better performance. Unfortunately, considering the cost, I did not continue to try. 100 times the data pre-training costs about $64

## [ADFWI](https://github.com/liufeng2317/ADFWI)



You can refer to their paper for detailed methods. It is a bit like TTT (test time training) in the ARC competition, where each sample is trained during the test. I think this is a general post-processing solution that can reduce your neural network prediction to a local optimum to achieve a certain score improvement. For me, LB was optimized from 22.1 to 17.0. 

## Waveform loss function

https://github.com/liufeng2317/ADFWI/blob/bv1.1/ADFWI/fwi/misfit/Envelope.py

For the loss functions Envelope loss and correlation loss that measure the difference between waveforms, the theoretically correct approach is to put the time step in the last dimension, but I found that putting the sensor in the last dimension works better.
