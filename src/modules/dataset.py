import numpy as np
import torch
from torch.utils.data import Dataset
from modules.utils import get_img

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CassavaDataset(Dataset):
    def __init__(self, df, data_root,
                 img_size,
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.img_size = img_size
        self.transforms = transforms
        self.data_root = data_root
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df['label'].max()+1)[self.labels]
                #print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        # get labels
        if self.output_label:
            target = self.labels[index]

        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(self.img_size, lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.img_size[0] * self.img_size[1]))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]

        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img
