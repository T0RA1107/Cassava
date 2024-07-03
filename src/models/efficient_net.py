from torch import nn
import timm

class CassvaImgClassifier(nn.Module):
    def __init__(self, n_class, dropout_rate=0.1, pretrained=False):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(n_features, n_class, bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
