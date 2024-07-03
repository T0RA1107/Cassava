import os
os.environ["WANDB_DIR"] = os.getcwd()
import sys
sys.path.append('src')
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler
from hydra import initialize_config_dir, compose
import wandb
import omegaconf

from models.get_model import get_model

from modules.utils import seed_everything
from modules.scheduler import CosineAnnealingWithWarmUp
from modules.loss import MyCrossEntropyLoss
from modules.train_manager import TrainManager
from train import prepare_dataloader, train_one_epoch, valid_one_epoch

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp"
    )
    parser.add_argument(
        "model_arch"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true"
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    return parser.parse_args()

def main():
    args = get_arguments()
    initialize_config_dir(config_dir=os.getcwd()+"/experiment/config", version_base=None)
    CFG = compose(config_name=args.exp + ".yaml")
    CFG = omegaconf.OmegaConf.to_container(
        CFG, resolve=True, throw_on_missing=True
    )

    if args.use_wandb:
        wandb.init(
            name=CFG['exp_name'],
            config=CFG,
            project=CFG['project'],
            job_type="training",
        )

    model_arch = args.model_arch
    os.makedirs("./weight/{}".format(model_arch), exist_ok=True)
    weight_path = "./weight/{}/{}".format(model_arch, CFG['exp_name'])
    os.makedirs("./weight/{}/{}".format(model_arch, CFG['exp_name']), exist_ok=True)

    if args.verbose:
        try:
            with open(os.getcwd()+"/experiment/log/" + args.exp + ".log", mode='x') as f:
                f.write("")
        except FileExistsError:
            pass


    seed_everything(CFG['common']['seed'])
    train = pd.read_csv('./data/train.csv')

    folds = StratifiedKFold(n_splits=CFG['common']['fold_num'], shuffle=True, random_state=CFG['common']['seed']).split(np.arange(train.shape[0]), train.label.values)
    device = torch.device(CFG['common']['device'])

    for fold, (trn_idx, val_idx) in enumerate(folds):
        # we'll train fold 0 first
        # if fold > 0:
        #     break

        train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, CFG[model_arch]['img_size'], CFG, trn_one_hot_label=True, data_root='./data/train_images/')


        model = get_model(model_arch)(train.label.nunique(), dropout_rate=CFG[model_arch]['dropout_rate'], pretrained=True).to(device)
        scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG[model_arch]['lr'], weight_decay=CFG[model_arch]['weight_decay'])
        scheduler = CosineAnnealingWithWarmUp(optimizer, CFG[model_arch]['warmup_epoch'], CFG[model_arch]['epochs'], CFG[model_arch]['start_lr'], CFG[model_arch]['warm_lr'], CFG[model_arch]['end_lr'])
        manager = TrainManager()

        loss_tr = MyCrossEntropyLoss(weight=None, smooth_factor=CFG[model_arch]['smooth_factor']).to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(CFG[model_arch]['epochs']):
            running_loss_train = train_one_epoch(epoch, model, loss_tr, scaler, optimizer, train_loader, device, CFG, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                running_loss_valid, acc_val, log = valid_one_epoch(epoch, model, loss_fn, val_loader, device, CFG, scheduler=None, schd_loss_update=False)

            torch.save(model.state_dict(),'{}/fold_{}_{}.pth'.format(weight_path, fold, epoch))
            manager.check_and_save_weight(running_loss_train, running_loss_valid)

            if args.use_wandb:
                wandb.log(
                    {
                        f"fold {fold}": {
                            f"lr": optimizer.param_groups[0]["lr"],
                            f"accuracy": acc_val,
                            f"loss/train": running_loss_train,
                            f"loss/valid": running_loss_valid,
                        }
                    },
                    step=epoch
                )
            if args.verbose:
                with open(os.getcwd()+"/experiment/log/" + args.exp + ".log", mode="a") as f:
                    f.write(f"epoch: {epoch}\n{log}\n")

        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
