# 38th Solution & How I think I survived the shake-up

Competition: vsb-power-line-fault-detection
Rank: #38
Source: https://www.kaggle.com/c/vsb-power-line-fault-detection/discussion/85186#latest-497311

Hello everybody!

**The Start :**
I joined this comptetion approx. a month ago, at the start, my main idea was to use scipy.signal to extract features about the peaks, after applying wavelet denoising. I first went with LSTMs because of the hype around high scoring public kernels, but could not get close to 0.7 LB (public).

**Downfalls about public Kernels :**
Here are a few things I believe were overused and caused the shaked_up.
- Keras CuDNNLSTMs (because of the randomness), move to PyTorch !
- Threshold selection for MCC optimization. It only caused overfitting on your oof data
- Checkpoints while training your model. Again, you're overfitting on your oof data
- Overall, I went with the idea that RNNs were predicting noise only. 

Not sure about the last point, because I've got one of my very first submissions that got 0.674 private with the ideas above.

**To the magical submission : CV 0.659 - Public 0.652 - Private 0.667**

Yes that's pretty stable, that's why I trusted this model more than others.

Feature Engineering : 
- Wavelet Denoizing
- "Bucketted" distribution features such as in the public kernels (exactly the same actually).
- The three phases were fed together in the network, BUT as three phases of an image. My input shape was (3, 160, nb_features). I used the most common target among the three phases as the label.

I tried all the features I came up with, these ones achieved the best CV, I don't know why.

**Model :** 
As my inputs were similar to images now, I went with an architecture similar to AlexNet, but with less parameters. The architecture looks like this approx. (I changed it a bit, not sure if this is the one that got me the best score) :

```
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 6 * 6, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(32, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 32 * 6 * 6)
        x = self.classifier(x)
        return x
```
100 epochs, 5 folds CV, batch_size 128, Learning rate 0.001

I worked on my gaming laptop which has 16Gb of ram and a 1060, so the 800000 long signals were quite a pain in the a** but I dealt with it.

**Final Word**
This competition was hard because you could not trust your LB, and you could only trust your CV if it achieved results similar to LB ones. Well that's what I understood at least.

Thanks everyone for participating, I'm waiting to read about others' solutions.!
