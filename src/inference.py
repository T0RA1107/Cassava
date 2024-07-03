from tqdm import tqdm
import numpy as np
import torch

def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()

        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]


    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
