# 2nd Place Solution - UBC-OCEAN

Competition: UBC-OCEAN
Rank: #2
Source: https://www.kaggle.com/c/UBC-OCEAN/discussion/465410

# Preface
The most significant difficulty in whole slide image (WSI) classification is the extremely high resolution, which should have been experienced by all competitors. Although the organizers of the competition provided a data type difficult to process, fortunately, the resolution of the data is much lower than that of typical WSI. In this discussion, we will provide a detailed introduction to our method.

# Overview
Following the commonly used methods in academia, we toke the following steps:
1. **Crop** an entire WSI into thousands of **patches**;
2. Use extractors to **extract the features**;
3. Train the **MIL** models.

# External Data
We used two external data with labels. All competitors can download without payment. We found that although more external data and Other class were used for training, there was no significant improvement in scores. We believe this is due to quality issues with external data or significant differences from competition data. Just as some competitors can achieve high scores without using external data, we believe that the external data is not necessary in this competition.
- https://wirtualnymikroskop.mostwiedzy.pl/list/
- https://www.cancerimagingarchive.net/collection/ptrc-hgsoc/

# Crop Patches and Extract Features
We create one Dataset for one WSI. Code is here:
```python
class SingleWSIDataset(Dataset):
    def __init__(self, data_path: str, wsi_name: str, patch_size: int, mode: str):
        super().__init__()
        self.data_path = data_path
        self.wsi_name = wsi_name
        self.ratio = ratio
        assert mode in ['train', 'test']
        self.mode = mode
        self.wsi = pyvips.Image.new_from_file(os.path.join(data_path, f'{mode}_images', wsi_name + '.png'))
        self.is_tma = self.wsi.height < 5000 and self.wsi.width < 5000
        self.patch_size = patch_size
        self.transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True), T.Normalize(mean=[0.2585, 0.2556, 0.2506], std=[0.229, 0.224, 0.225])])
        self.cor_list = self.get_patch()

    def get_patch(self):
        cor_list = []
        if self.is_tma:
            thumbnail = self.wsi
        else:
            thumbnail = pyvips.Image.new_from_file(os.path.join(self.data_path, f'{self.mode}_thumbnails', self.wsi_name + '_thumbnail.png'))
        wsi_width, wsi_height = self.wsi.width, self.wsi.height
        thu_width, thu_height = thumbnail.width, thumbnail.height
        h_r, w_r = wsi_height / thu_height, wsi_width / thu_width
        down_h, down_w = int(self.patch_size / h_r), int(self.patch_size / w_r)
        cors = [(x, y) for y in range(0, thu_height, down_h) for x in range(0, thu_width, down_w)]
        for x, y in cors:
            tile = thumbnail.crop(x, y, min(down_w, thu_width - x), min(down_h, thu_height - y)).numpy()[..., :3]
            black_bg = np.mean(tile, axis=2) < 20
            tile[black_bg, :] = 255
            mask_bg = np.mean(tile, axis=2) > 235
            if np.sum(mask_bg) < min(down_h, thu_height - y) * min(down_w, thu_width - x) * 0.7 or len(cor_list) == 0 or self.is_tma:
                cor_list.append((int(x * w_r), int(y * h_r)))
        if self.is_tma:
            return cor_list
        if self.wsi.height < 40000 and self.wsi.width < 40000:
            R_ratio = 0.8
        elif self.wsi.height < 80000 and self.wsi.width < 80000:
            R_ratio = 0.6
        else:
            R_ratio = 0.5
        random.shuffle(cor_list)
        cor_list = cor_list[:max(int(len(cor_list) * R_ratio), 1)]
        return cor_list

    def __len__(self):
        return len(self.cor_list)

    def __getitem__(self, idx):
        x, y = self.cor_list[idx]
        tile = self.wsi.crop(x, y, min(self.patch_size, self.wsi.width - x), min(self.patch_size, self.wsi.height - y)).numpy()[..., :3]
        tile = self.transform(tile)
        return tile
```
# Feature Extraction Model
We used **dino_vit_small_patch16_200ep.torch** and **dino_vit_small_patch8_200ep.torch**.
- https://github.com/lunit-io/benchmark-ssl-pathology/releases/tag/pretrained-weights
# MIL Model
- ABMIL
- DSMIL
- TransMIL
# Codes
Simplified Version
- https://www.kaggle.com/code/zznznb/wsi-train
- https://www.kaggle.com/code/zznznb/wsi-inference-public-0-6-private-0-58

Final Version
- https://www.kaggle.com/code/hustzx/2nd-0-61-train-abmil-dsmil-transmil
- https://www.kaggle.com/code/hustzx/2nd-0-61-infernece-abmil-dsmil-transmil

Feature Extraction Codes
- https://github.com/ZeningZeng/UBC-OCEAN
