from moduli.data import SkinDataset
from moduli.data.utils import normalize_dataset
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

seed=42
np.random.seed(seed)
# Import images as dataset
data_folder = "../../sint-data-noSBD-noDouble"
data = datasets.ImageFolder(root=data_folder)
all_path = [s[0] for s in data.samples]
all_label = [s[1] for s in data.samples]
print(f"Total images: {len(all_path)}")

# Split the images in 80% training, 10% validation and 10% testing
train_size = 0.8
train_path, temp_path, train_label, temp_label = train_test_split(all_path, all_label, train_size=train_size, stratify=all_label, random_state=seed)

val_size = 0.5
val_path, test_path, val_label, test_label = train_test_split(temp_path, temp_label, train_size=val_size, stratify=temp_label, random_state=seed)

# Compute mean and std of the dataset
mean = None
std = None
#mean, std = normalize_dataset(train_path, crop_size=224)
if mean is None and data_folder[6:10]=="data":
    mean = [0.60045856, 0.44390684, 0.40402648]
    std = [0.22650158, 0.2031579, 0.21181032]
    print("Using mean and std from original dataset (for seed 42)")
elif mean is None and data_folder[6:10]=="sint":
    mean = [0.6147956, 0.4593298, 0.4273037]
    std = [0.228707, 0.20079376, 0.20833343]
    print("Using mean and std from synthetic dataset (for seed 42)")
elif mean is None:
    print("Unknown dataset.")
    raise Exception("Dataset unknown. Set mean and std manually.")

#Create train, validation and test dataset with transforms for data augmentation
transform = transforms.Compose([transforms.RandomRotation(degrees=90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor()
#    transforms.Normalize(mean=mean, std=std)
])

train_set = SkinDataset(paths=train_path, labels=train_label, transform=transform, augm=True, selec_augm=False)
print(f"Training images: {len(train_path)}")
val_set = SkinDataset(paths=val_path, labels=val_label, transform=transform, augm=False)
print(f"Validation images: {len(val_path)}")
test_set = SkinDataset(paths=test_path, labels=test_label, transform=transform, augm=False)
print(f"Test images: {len(test_path)}")
#raise Exception("Stopped")

# DataLoader for each set
size_batch = 128 if torch.cuda.is_available() else 16
print(f"Batch size: {size_batch}")
train_loader = DataLoader(train_set, batch_size=size_batch, shuffle=True)
val_loader = DataLoader(val_set, batch_size=size_batch, shuffle=False)
test_loader = DataLoader(test_set, batch_size=size_batch, shuffle=False)

# Initialize server model and variables for training process
n_classes = 6
fed_server = EffNetB0(n_classes=n_classes, weights=EfficientNet_B0_Weights.DEFAULT)
nome = "presint-nonorm"
model_dir = f"modelli/{nome}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
fed_server = fed_server.to(device)
print()

# What to perform in the script (in order)
finetune_train = True
opt_fine = optim.AdamW(fed_server.parameters(), lr=0.005)

test1 = True

training = True
opt = optim.AdamW(fed_server.parameters(), lr=0.0001)

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
        epochs=10,
        early_stop=5,
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
    fed_server.load_state_dict(torch.load(f"{model_dir}/best_{nome}.pt", map_location=torch.device(device)))
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
    fed_server.load_state_dict(torch.load(f"{model_dir}/best_{nome}.pt", map_location=torch.device(device)))
    trainer.train_this_model(
        model=fed_server,
        train_loader=train_loader,
        val_loader=val_loader,
        opt=opt,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        epochs=30,
        early_stop=5,
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
    fed_server.load_state_dict(torch.load(f"{model_dir}/best_{nome}.pt", map_location=torch.device(device)))
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
