import torch.nn as nn
from torchvision.models import efficientnet_b0

class EffNetB0(nn.Module):
    """Create the NN model"""
    def __init__(self, n_classes: int=6, weights: list=None):
        # Create the class Net
        super(EffNetB0, self).__init__()
        # Model is EfficientNet B0
        self.model = efficientnet_b0(weights=None, progress=False)
        # Set number of classes as output of classifier
        b0_out = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=b0_out, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=n_classes)
        )
        if weights is not None:
            param_names = [name for name in self.model.state_dict.keys()]
            state_dict = {}
            for name, array in zip(param_names, weights):
                state_dict[name] = torch.from_numpy(array)
            
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
        
    def extract_backbone_features(self, x, layer_idx=None):
        if layer_idx is None:
            return self.model.features(x)
        
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if layer_idx is not None and idx == layer_idx:
                break
        return x
