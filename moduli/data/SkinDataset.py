import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SkinDataset(Dataset):
    def __init__(self,
        paths: list,
        labels: list,
        transform: T.Compose,
        augm: bool=False,
        selec_augm: bool=False):

        self.image_paths = paths
        self.labels = labels
        self.augm = augm
        self.selec_augm = selec_augm
        # Last three transforms (CenterCrop(224), ToTensor and Normalize) are always needed
        self.base_tr = T.Compose(transform.transforms[-3:])
        self.transform = transform
        if self.selec_augm:
            self.low_samples_classes = ["actinic keratosis", "seborrheic keratosis", "squamous cell carcinoma"]

        self.targets = self.labels
        self.classes = ["actinic keratosis", "basal cell carcinoma", "melanoma", "nevus", "seborrheic keratosis", "squamous cell carcinoma"]
        self.class_to_idx = {
            0: "actinic keratosis",
            1: "basal cell carcinoma",
            2: "melanoma",
            3: "nevus",
            4: "seborrheic keratosis",
            5: "squamous cell carcinoma"}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        if self.augm==False:
            image = self.base_tr(image)
        elif self.selec_augm:
            if self.class_to_idx[label] in self.low_samples_classes:
                image = self.transform(image)
            else:
                image = self.base_tr(image)
        else:
            image = self.transform(image)
           
        label = torch.tensor(label, dtype=torch.long)
        return image, label