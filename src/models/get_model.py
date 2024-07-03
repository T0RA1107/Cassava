from .efficient_net import CassvaImgClassifier
from .vit import CassvaImgClassifierViT
from .deit import CassvaImgClassifierDeiT

def get_model(model_arch):
    if model_arch == 'efficient_net':
        return CassvaImgClassifier
    elif model_arch == 'vit':
        return CassvaImgClassifierViT
    elif model_arch == 'deit':
        return CassvaImgClassifierDeiT
    else:
        raise NotImplemented
