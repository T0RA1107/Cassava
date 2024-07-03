from torch import nn
import timm

class CassvaImgClassifierDeiT(nn.Module):
    def __init__(self, n_class, dropout_rate=0.1, pretrained=True):
        super().__init__()

        self.model = timm.create_model('deit3_base_patch16_384', pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
