from moduli.data import SkinDataset
from moduli.learning import trainer, tester
from moduli.NNmodel import EffNetB0
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.models import EfficientNet_B0_Weights
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Import images as dataset
data_folder = "../../sint-data-noSBD-noDouble"
data = datasets.ImageFolder(root=data_folder)
all_path = [s[0] for s in data.samples]
all_label = [s[1] for s in data.samples]
print(f"Total images: {len(all_path)}")

# Split the images in 80% training, 10% validation and 10% testing
train_size = 0.8
train_path, temp_path, train_label, temp_label = train_test_split(all_path, all_label, train_size=train_size, stratify=all_label, random_state=26)

val_size = 0.5
val_path, test_path, val_label, test_label = train_test_split(temp_path, temp_label, train_size=val_size, stratify=temp_label, random_state=26)

#Create train, validation and test dataset with transform for data augmentation
transform = [transforms.RandomRotation(degrees=90),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
    transforms.RandomVerticalFlip()
]

train_set = SkinDataset(paths=train_path, labels=train_label, transform=transform, augm_type="all")
print(f"Training images: {len(train_path)}")
val_set = SkinDataset(paths=val_path, labels=val_label, transform=transform, augm_type="all")
print(f"Validation images: {len(val_path)}")
test_set = SkinDataset(paths=test_path, labels=test_label, transform=transform, augm_type="all")
print(f"Test images: {len(test_path)}")

# DataLoader for each set
size_batch = 128 if torch.cuda.is_available() else 16
print(f"Batch size: {size_batch}")
train_loader = DataLoader(train_set, batch_size=size_batch, shuffle=True)
val_loader = DataLoader(val_set, batch_size=size_batch, shuffle=False)
test_loader = DataLoader(test_set, batch_size=size_batch, shuffle=False)

# Initialize server model and variables for training process
n_classes = 6
fed_server = EffNetB0(n_classes=n_classes, weights=EfficientNet_B0_Weights.DEFAULT)
nome = "pre-sint"
model_dir = f"modelli/{nome}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
fed_server = fed_server.to(device)
print()

# What to perform in the script (in order)
finetune_train = True
opt_fine = optim.Adam(fed_server.parameters(), lr=0.01)

test1 = True

training = True
opt = optim.Adam(fed_server.parameters(), lr=0.0005)

test2 = True

# Fine tuning of classifier or loading previous model based on its name
train_loss = []
train_acc = []
val_acc = []
val_loss = []
if finetune_train:
    trainer.train_this_model(model=fed_server,
        train_loader=train_loader,
        val_loader=val_loader,
        opt=opt_fine,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        epochs=15,
        early_stop=4,
        name=nome,
        fine_tune=True
    )
else:
    try:
        fed_server.load_state_dict(torch.load(f"{model_dir}/best_{nome}.pt", map_location=torch.device(device)))
        print(f"Previous model {nome} loaded.")
        train_acc, train_loss, val_acc, val_loss = trainer.read_train_results(model_dir)
    except:
        print("Previous model not found.")

# Testing results: default model with trained classifier
test_preds = []
test_true = []
if test1:
    test_preds, test_true = tester.test_this_model(
        model=fed_server,
        test_loader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        iter=0,
        report=True,
        name=nome
    )    
else:
    print("Test skipped.")
    
# Training the whole model or loading previous model based on its name
if training:
    trainer.train_this_model(
        model=fed_server,
        train_loader=train_loader,
        val_loader=val_loader,
        opt=opt,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        epochs=15,
        early_stop=4,
        name=nome,
        fine_tune=False
    )
else:
    try:
        fed_server.load_state_dict(torch.load(f"{model_dir}/best_{nome}.pt", map_location=torch.device(device)))
        print(f"Previous model {nome} loaded.")
        
    except:
        print("Previous model not found.")
        
# Testing the whole model
test_preds = []
test_true = []
if test2:
    test_preds, test_true = tester.test_this_model(
        model=fed_server,
        test_loader=test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        iter=0,
        name=nome,
        report=True
    )
    
else:
    print("Test skipped.")
