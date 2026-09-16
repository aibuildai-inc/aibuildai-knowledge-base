# 7th Place Solution（simple but messy）

Competition: prostate-cancer-grade-assessment
Rank: #7
Source: https://www.kaggle.com/c/prostate-cancer-grade-assessment/discussion/169225

First of all, Thank you very much to organizers.

This challenge is very similar to APTOS-2019 which I have worked for months as a course assignment. So I simlpy used the pipeline of my course assignment(Based on [Lex Toumbourou‘s solution](https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/107947), thanks a lot) with some revised details. Also thanks a lot to [Iafoss](https://www.kaggle.com/iafoss/panda-16x128x128-tiles) and [Qishen Ha](https://www.kaggle.com/haqishen/train-efficientnet-b0-w-36-tiles-256-lb0-87) for their useful notebooks.

Our model is simple but messy.

## Tiles
We tried 256x256x32, 192x192x64, 154x154x100 but they didn't show some difference on Public LB. 

We also propoesd a new tile approach. It can contain more pathological parts without destroying the shape features. Large size tile ensures that the shape features will not be lost, while small size tiles ensure that the blank area is not that large.

    def get_tiles_combine(img,mode=0):
        images = np.ones((1536, 1536, 3))*255
        h, w, c = img.shape
        result_all=[]
        pad_h = (256 - h % 256) % 256 + ((256 * mode) // 2)
        pad_w = (256 - w % 256) % 256 + ((256 * mode) // 2)
        #print(pad_h,pad_w,c)
        img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], 'constant',constant_values=255)
        windows=[256,256,256,256,192,192,128]
        x_start=0
        for i in range(len(windows)):
            result = []
            window_size=windows[i]
            for x in range((h+pad_h)//window_size):
                for y in range((w+pad_w)//window_size):
                    tile=img2[x*window_size:(x+1)*window_size,y*window_size:(y+1)*window_size]
                    result.append([x,y,tile.sum()])
            #print(len(result))
            result.sort(key=lambda ele:ele[2])
            result=result[:1536//window_size]
            #print(len(result),result)
            for y in range(min(1536//window_size,len(result))):
                xx=result[y][0]
                yy=result[y][1]
                result_all.append([xx,yy])
                images[x_start:x_start+window_size,y*window_size:(y+1)*window_size]=\
                    img2[xx*window_size:(xx+1)*window_size,yy*window_size:(yy+1)*window_size].copy()
                img2[xx*window_size:(xx+1)*window_size,yy*window_size:(yy+1)*window_size]=255
            x_start=x_start+windows[i]
        return images 
 
##Models

We simply used Efficientnet-B0. We tried B1-B3,Densenet and Resnext, but they didn't show some difference on Public LB and need more GPU memory.

According to APTOS-2019, we used the GeM pooling:


    def gem(x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    class GeM(nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super(GeM,self).__init__()
            self.p = Parameter(torch.ones(1)*p)
            self.eps = eps
        def forward(self, x):
            return gem(x, p=self.p, eps=self.eps)       
        def __repr__(self):
            return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

We also designed a more complex model based on b0. It has lower Public LB(average about 0.85) so we didn't add it to our final models. But it got the highest single-model Private LB (max 0.926, average about 0.920, What a pitty!). We will do some more experiments on this model.

##Loss and label

BCE Loss and label smoothing: 3-&gt;[0.95,0.95,0.95,0.95,0.05,0.05]

We also tried regression with mse loss and smooth L1 loss, but they didn't show any improvement.

##Ensemble
8 models with 6 * TTA:

&gt; 1: fold_1 b0 256-tile  Public LB:0.879, Private LB:0.904.

&gt; 2: fold_3 b0 256-tile Public LB:0.879, Private LB:0.899.

&gt; 3: fold_4 b0 256-tile Public LB:0.886, Private LB:0.883.

&gt; 4: fold_4 b0 combine-tile Public LB:0.879, Private LB:0.910.

&gt; 5: fold_4 b0 256-tile Public LB:0.880, Private LB:0.909.

&gt; 6: fold_4 b0 256-tile Public LB:0.891, Private LB:0.920.

&gt; 7: fold_4 b0 combine-tile Public LB:0.881 Private LB:0.917.

&gt; 8: fold_0 b0 256-tile Public LB:0.872, Private LB:0.906.


The final model has Public LB:0.894, Private LB:0.932.
