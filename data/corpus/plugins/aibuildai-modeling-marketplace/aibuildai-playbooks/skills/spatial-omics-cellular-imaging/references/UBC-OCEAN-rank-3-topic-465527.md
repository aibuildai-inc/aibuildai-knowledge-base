# 3rd Place Solution for the UBC-OCEAN UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)

Competition: UBC-OCEAN
Rank: #3
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465527

## Context

Business context: [UBC Ovarian Cancer Subtype Classification and Outlier Detection](https://www.kaggle.com/competitions/UBC-OCEAN)

Data context: [The] challenge in this competition is to classify the type of ovarian cancer from microscopy scans of biopsy samples. [Link to data description](https://www.kaggle.com/competitions/UBC-OCEAN/data)

## Overview of the Approach

- Finding **more public external data was key** for me. Overfitting was a big problem due to the small number of examples. Initially I hoped the [CLAM](https://github.com/mahmoodlab/CLAM) or a multiple instance learning (MIL) approach could remedy that, because many images are so large they can be split in tens of thousands of tiles. But my models were still overfitting a lot. I assume the tiles of the same patient are similar in important ways and the models can use that as shortcuts that don't generalize well. Or there is just so much heterogeniety between cancers, so that the number of samples are just not enough to capture all different variants of the subtypes.
- I used the **segmentation data provided to create synthetic tumor micro array (TMA) images**, basically jsut cropping tiny images from the segmentation of the large image. I did this for the cancer tissue and generated some "Other" synthetic images by cropping small tiles that were marked as healthy or as stroma.
- Following the paper "[A Good Feature Extractor Is All You Need for Weakly Supervised Learning in Histopathology](https://arxiv.org/pdf/2311.11772.pdf)" I used the **pretrained model Lunit-DINO to extract smaller size features** to handle the massive image size. I ran the feature extractino in 16-bit so that it runs faster. I didn't see much negative impact on feature quality.

- **I filtered the tiles containing tissue using the thumbnails and then cropped the tissue tiles using PyVips**. I lost so much time on this one, because I first tried to re-write the feature extraction code in CLAM, but couldn't make it work with the resource limits on Kaggle. Then I tried using the [large_image](https://github.com/girder/large_image) library that is made for histopatholy images, but could not make it work with the Kaggle resource limits. I alternated between out of memory, out of disk and timeouts. Finally I made it work using PyVips and asynchronous data loading in PyTorch. This part of the competition was frustrating, as I wanted to spend my time on deep learning and spend weeks on image processing and guessing Kaggle errors.
- **On the extracted features I trained the [CLAM](https://github.com/mahmoodlab/CLAM) model**, which is simiar to MIL, but calculates an attention matrix to weight the tiles. I made some changes to the instance level loss function for the "Other" label, as it's sematically different from the other labels. A tile of a slide containing a cancer subtype could still be have the label "Other", if the tile would only show healthy tissue.

## **Details of the submission**
### CLAM Model

The following diagram shows the CLAM model from [Mahmood Lab @ Harvard/BWH & MGH ](https://faisal.ai/)[1]. This model takes as input concatenated features that have been extracted from all tiles of a Whole Slide Image containing tissue. 

The top part of the diagram calculates the attention scores A, a vector with one entry per tile. The bottom part creates an A-weighted sum of the transformed input features h and feeds it into a multi-class classification head.

[CLAM diagram]

Diagram from Paul Pham [2]


PyTorch Code for my adapted CLAM model:

```Python
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = 0, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()
        ]
        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()
        ]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", n_classes=2, dropout = 0, k_sample=8,
            instance_loss_fn=None, subtyping=False, feature_dim=1024, use_inst_predictions=True,
            label_mapping=None, class_weights=None, inst_class_depth=None, inst_dropout=None):
        super().__init__()
        self.size_dict = {
            "very small": [feature_dim, 256, 128],
            "small": [feature_dim, 512, 256],
            "big": [feature_dim, 1024, 512],
            "xl": [feature_dim, 2048, 1024]
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout > 0:
            fc.append(nn.Dropout(dropout))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = []  
        for class_idx in range(n_classes):
            layers = []
            for depth_idx in range(inst_class_depth-1):
                divisor = 2 ** depth_idx        
                layers.append(nn.Linear(size[1] // divisor, size[1] // (divisor * 2)))
                layers.append(nn.ReLU())
                if inst_dropout is not None:
                    layers.append(nn.Dropout(inst_dropout))
            layers.append(nn.Linear(size[1] // 2**(inst_class_depth-1), 1))
            instance_classifiers.append(nn.Sequential(*layers))  
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.use_inst_predictions = use_inst_predictions
        self.other_idx = label_mapping['Other']
        self.class_weights = class_weights
        initialize_weights(self)
        self.to('cuda')

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).float()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).float()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier, is_tma, is_other_class): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        
        if is_tma:
            k_sample = self.k_sample // 2
        else:
            k_sample = self.k_sample

        if k_sample <= math.ceil(A.shape[1] / 2):
            top_p_ids = torch.topk(A, k_sample)[1][-1] # [1][-1] selects the last index
        else:
            top_p_ids = torch.topk(A, math.ceil(A.shape[1] / 2))[1][-1]
            top_p_ids = top_p_ids.repeat(k_sample)[:k_sample]
        top_p = torch.index_select(h, dim=0, index=top_p_ids) # dim = k_sample x self.size_dict[1]
        if k_sample <= math.ceil(A.shape[1] / 2):
            top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        else:
            top_n_ids = torch.topk(-A, math.ceil(A.shape[1] / 2))[1][-1]
            top_n_ids = top_n_ids.repeat(k_sample)[:k_sample]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        # errors get evaluated on positive and negative labels, to also constain A for low attention
        # on negative tiles, but for prediction we only care about the top tiles.
        p_logits = classifier(top_p) # dim = k_sample
        n_logits = classifier(top_n)
        inst_preds = (p_logits.squeeze() > 0).long()
        # we give more weight to the positive targets as we have more negative targets.
        p_loss = self.instance_loss_fn(p_logits.squeeze(), p_targets) * (self.n_classes -1)
        n_loss = self.instance_loss_fn(n_logits.squeeze(), n_targets)
        if not is_tma and not is_other_class:
            loss = p_loss + n_loss
        else: loss = p_loss
        return loss, inst_preds, p_targets, p_logits
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier, is_tma):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if is_tma:
            k_sample = self.k_sample // 2
        else:
            k_sample = self.k_sample

        # we allow at max the top half of the bag to be selected, 
        # otherwise we repeat the top half.
        if k_sample <= math.ceil(A.shape[1] / 2):
            top_ids = torch.topk(A, k_sample)[1][-1]
        else:
            top_ids = torch.topk(A, math.ceil(A.shape[1] / 2))[1][-1]
            top_ids = top_ids.repeat(k_sample)[:k_sample]
        top_inst = torch.index_select(h, dim=0, index=top_ids)
        top_targets = self.create_negative_targets(k_sample, device)
        

        logits = classifier(top_inst)
        inst_preds = (logits.squeeze() > 0).long()
        instance_loss = self.instance_loss_fn(logits.squeeze(), top_targets)
        return instance_loss, inst_preds, top_targets, logits
    

    def forward(self, h, bag_pred_weight:float, is_tma:bool, label=None, attention_only=False):
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h) # shape 1 x self.size_dict[1]
        logits = self.classifiers(M)
        bag_Y_prob = F.softmax(logits.squeeze(), dim=0)

        if is_tma:
            k_sample = self.k_sample // 2
        else:
            k_sample = self.k_sample

        all_inst_logits = []
        top_p_ids = None
        if bag_pred_weight < 1 and label is not None:        
            total_inst_loss = 0.0
            all_inst_preds = []
            all_targets = []
            for i in range(len(self.instance_classifiers)):
                classifier = self.instance_classifiers[i]
                if i == label.item(): #in-the-class:
                    is_other_class = (label.item() == self.other_idx)
                    instance_loss, inst_preds, targets, inst_logits = self.inst_eval(A, h, classifier, is_tma, is_other_class)
                    all_inst_preds.extend(inst_preds.cpu().numpy())                 
                    all_targets.extend(targets.cpu().numpy())
                    #class_probs = F.softmax(inst_logits, dim=1)
                    all_inst_logits.append(inst_logits)
                    if self.class_weights is not None:
                        instance_loss *= self.class_weights[i]
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, inst_preds, targets, inst_logits = self.inst_eval_out(A, h, classifier, is_tma)
                        all_inst_preds.extend(inst_preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())              
                        all_inst_logits.append(inst_logits)
                    else:
                        continue
                
                total_inst_loss += instance_loss 

            if self.subtyping:
                # the 2 corrects for the upscaling of the positive instance loss
                total_inst_loss /= 2 * len(self.instance_classifiers)
        else:
            if self.k_sample <= math.ceil(A.shape[1] / 2):
                top_p_ids = torch.topk(A, k_sample)[1][-1] # [1][-1] selects the last index
            else:
                top_p_ids = torch.topk(A, math.ceil(A.shape[1] / 2))[1][-1]
                top_p_ids = top_p_ids.repeat(k_sample)[:k_sample]
            top_p = torch.index_select(h, dim=0, index=top_p_ids)
            for classifier in self.instance_classifiers:
                class_logits = classifier(top_p)
                all_inst_logits.append(class_logits)

                
        if self.use_inst_predictions: 
            all_inst_logits = torch.concatenate(all_inst_logits, axis=1)  # dim k_sample x n_classes
            # take probs of all k_sample classifiers and turn them into multi-class probabilities
            # weight tile instance predictions by A
            if self.k_sample <= math.ceil(A.shape[1] / 2):
                top_p_ids = torch.topk(A, k_sample)[1][-1] # [1][-1] selects the last index
            else:
                top_p_ids = torch.topk(A, math.ceil(A.shape[1] / 2))[1][-1]
                top_p_ids = top_p_ids.repeat(k_sample)[:k_sample]

            all_inst_logits =A_raw[0, top_p_ids].reshape(-1, 1) * all_inst_logits
            softmax_inst_probs = torch.softmax(all_inst_logits, dim=1)
            agg_inst_probs = softmax_inst_probs 
            agg_inst_probs = torch.mean(agg_inst_probs, dim=0) # result has dim k_sample
            Y_probs = bag_Y_prob * bag_pred_weight + agg_inst_probs * (1 - bag_pred_weight)
        Y_hat = torch.topk(Y_probs, 1, dim=0)[1]
        
        results_dict = {}
        if bag_pred_weight < 1:
            results_dict.update({
                'all_inst_logits': all_inst_logits.detach().cpu().numpy(),
                'agg_inst_probs': agg_inst_probs.detach().cpu().numpy()
            })
        if self.use_inst_predictions: 
            results_dict.update({
                'softmax_inst_probs': softmax_inst_probs.detach().cpu().numpy()
            })
        if label is not None:
            results_dict.update({
                'inst_labels': np.array(all_targets),
                'inst_preds': np.array(all_inst_preds).flatten(),
                'instance_loss': total_inst_loss
            })

        return logits, Y_probs, Y_hat, A_raw, results_dict
```



### Description of the Data Used

From the [Cancer Imaging Archive](https://www.cancerimagingarchive.net/) I used the [Ovarian Bevacizumab Response](https://doi.org/10.7937/TCIA.985G-EY35) and the [CPTAC-OV](https://doi.org/10.7937/TCIA.ZS4A-JD58) data. The labels didn't map perfectly to the labels of the competition. For example a label would be Papillary Serous Carcinoma, which I assumed means either HGSC or LGSC. I just used a model trained on the my other data to decide which of the two to select.

I used the Ovarian [Carcinoma Histopathology Dataset](https://www.medicalimageanalysis.com/data/ovarian-carcinomas-histopathology-dataset) by the [Hamarneh Lab](https://www.medicalimageanalysis.com/home).

I also used data from the [Stanford Tissue Microarray Database](https://tma.im/cgi-bin/home.pl), most of which didn't have H&E staining, but at least the cell shapes looked similar and I wanted to also have some TMA data.

I also got in touch with [Cooperative Human Tissue network](https://chtn.cancer.gov), who where so nice to allow me to use the data that was publicly available on the website. I thank them for the time they took to discuss this topic, even though I didn't use their data in the end, as they took the data down from their website after my outreach. As the data wasn't available for other participants anymore, I assume I'm not allowed to use it to train my model.

### Validation Setup

For a long time I pooled all my data and used 5-fold cross validation, making sure multiple images from the same patient would all be in the same fold. Still this lead to inflated validation scores. I seems like my models where overfitting by using some shortcuts from my datasets that do not generalize. Later I excluded the data from the Harmanreh lab completely for validation which lead to much more reliable cross-validation scores.

### Technical Setup

As the data size of the competition was so large I trained my models locally on my desktop computer with a GTX 4090 card. Feature extraction would take around 6 hours for all my datasets and training my model would take another hour.


### Data sources

A big thank you to everyone who made their pathological image data publicly available. This is a tremendous help for anyone who wants to build deep learning models to improve digital pathology, but especially for individuals without access to close source clinical data.

Bevacizumab: Wang et al. *Weakly Supervised Deep Learning for Prediction of Treatment Effectiveness on Ovarian Cancer from Histopathology Images.* Computerized Medical Imaging and Graphics. [https://doi.org/10.1016/j.compmedimag.2022.102093](https://gcc02.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdoi.org%2F10.1016%2Fj.compmedimag.2022.102093&data=05|01|kirbyju@mail.nih.gov|a1a4f263214846a156f908da58bd5e3c|14b77578977342d58507251ca2dc2b06|0|0|637919868422135484|Unknown|TWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D|3000|||&sdata=424GtHH8SDUEjvHtaXZvOwt9hcHaZgl36YHL1tzH6T4%3D&reserved=0)

CPTAC-OV: National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). (2020). **The Clinical Proteomic Tumor Analysis Consortium Ovarian Serous Cystadenocarcinoma Collection (CPTAC-OV)** (Version 3) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.ZS4A-JD58

Harmanreh Data: Köbel, Martin; Kalloger, Steve E.; Baker, Patricia M.; Ewanowich, Carol A.; Arseneau, Jocelyne; Zherebitskiy, Viktor; Abdulkarim, Soran; Leung, Samuel; Duggan, Máire A.; Fontaine, Dan; et al. (2010). "Diagnosis of ovarian carcinoma cell type is highly reproducible: a transcanadian study". *The American Journal of Surgical Pathology*, 34(7), 984–993. LWW.

I used some screenshots for healthy tissue from [The Human Protein Atlas](https://www.proteinatlas.org/learn/dictionary/normal/ovary) and the [University of Michigan Histology and Cirtual Microscopy](https://histology.medicine.umich.edu/resources/female-reproductive-system#ovary-oviduct-suggested-readings) page. I zoomed into their full section image of a healthy ovary, zoomed in to the maximum and took screenshots of many differently looking regions.


### Open Source Code

A second thank you to everyone who made their code or their model weights openly available. This greatly improves innovation and allows individual contributors to stand on the shoulders of giants.

[CLAM](https://github.com/mahmoodlab/CLAM): Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w

[PyVips](https://libvips.github.io/pyvips)

[HistomicsTK(https://github.com/DigitalSlideArchive/HistomicsTK)

[large_iamge]https://github.com/girder/large_image


### Research Papers

Wölflein, Georg; Ferber, Dyke; Meneghetti, Asier Rabasco; El Nahhas, Omar S. M.; Truhn, Daniel; Carrero, Zunamys I.; Harrison, David J.; Arandjelović, Ognjen; Kather, Jakob N. (2023). "A Good Feature Extractor Is All You Need for Weakly Supervised Learning in Histopathology". arXiv:2311.11772.


**Thanks for you interest in my solution and you can fine me on [Twitter here](https://twitter.com/swanint).**
