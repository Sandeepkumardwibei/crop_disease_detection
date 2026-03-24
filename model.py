import torch.nn as nn
import timm

class HybridPlantModel(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0
        )
        
        feature_dim = self.backbone.num_features
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        
        x = self.backbone(x)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = self.classifier(x)
        
        return x