import os
os.environ["WANDB_DIR"] = os.getcwd()
import sys
sys.path.append('src')
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from hydra import initialize_config_dir, compose
import omegaconf

from models.get_model import get_model

from modules.dataset import CassavaDataset
from modules.utils import seed_everything
from modules.transforms import get_inference_transforms
from inference import inference_one_epoch

def main():
    initialize_config_dir(config_dir=os.getcwd()+"/experiment/config", version_base=None)
    CFG = compose(config_name="plabel.yaml")
    CFG = omegaconf.OmegaConf.to_container(
        CFG, resolve=True, throw_on_missing=True
    )

    enssemble_label = []

    data = pd.DataFrame()
    data['image_id'] = list(os.listdir('./data/train_images'))

    device = torch.device(CFG['common']['device'])

    for model_arch in ['vit', 'deit']:

        img_size = (CFG[model_arch]['img_size'], CFG[model_arch]['img_size'])
        ds = CassavaDataset(data, './data/train_images', img_size, transforms=get_inference_transforms(img_size), output_label=False)

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=CFG['common']['valid_bs'],
            num_workers=CFG['common']['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        used_epochs = list(map(int, CFG[model_arch]['used_epochs'].split(' ')))

        for fold in range(CFG['common']['fold_num']):
            print('Inference fold {} started'.format(fold))
            model = get_model(model_arch)(5, dropout_rate=CFG[model_arch]['dropout_rate'], pretrained=False).to(device)
            model.load_state_dict(torch.load('{}/fold_{}_{}.pth'.format(CFG[model_arch]['model_weight_dir'], fold, used_epochs[fold])))

            with torch.no_grad():
                for _ in range(CFG['common']['tta']):
                    enssemble_label += [inference_one_epoch(model, data_loader, device)/CFG['common']['tta']]

            del model
            torch.cuda.empty_cache()

    enssemble_label = np.mean(enssemble_label, axis=0)
    np.save(CFG['common']['dst_file'], enssemble_label)

if __name__ == "__main__":
    main()
