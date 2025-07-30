import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
#from tqdm import tqdm

"""Function to save a checkpoit of a trained model"""
def save_checkpoint(filepath: str,
                    model: nn.Module,
                    name: str,
                    optimizer: optim.Optimizer,
                    epoch: int,
                    batch_idx: int,
                    loss: callable):
    torch.save({
        'epoch': epoch,
        'batch': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, filepath)


"""Function to load a checkpoit of a previously trained model"""
def load_checkpoint(filepath: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint.get('batch', 0)
    return model, optimizer, start_epoch, start_batch

def write_train_results(
    train_acc_evo: list,
    train_loss_evo: list,
    val_acc_evo: list,
    val_loss_evo: list,
    name: str="GiveMeName",
    fine_tune: bool=False):
    if fine_tune:  
        with open(f"modelli/{name}/train_loss_tune.txt", "w") as f:
            for loss in train_loss_evo:
                f.write(f"{loss}\n")
                    
        with open(f"modelli/{name}/train_acc_tune.txt", "w") as f:
            for acc in train_acc_evo:
                f.write(f"{acc}\n")
        
        with open(f"modelli/{name}/val_loss_tune.txt", "w") as f:
            for loss in val_loss_evo:
                f.write(f"{loss}\n")
                    
        with open(f"modelli/{name}/val_acc_tune.txt", "w") as f:
            for acc in val_acc_evo:
                f.write(f"{acc}\n")
    else:
        with open(f"modelli/{name}/train_loss.txt", "w") as f:
            for loss in train_loss_evo:
                f.write(f"{loss}\n")
                    
        with open(f"modelli/{name}/train_acc.txt", "w") as f:
            for acc in train_acc_evo:
                f.write(f"{acc}\n")
        
        with open(f"modelli/{name}/val_loss.txt", "w") as f:
            for loss in val_loss_evo:
                f.write(f"{loss}\n")
                    
        with open(f"modelli/{name}/val_acc.txt", "w") as f:
            for acc in val_acc_evo:
                f.write(f"{acc}\n")


"""Function to train a NN model"""
def train_this_model(
    model: nn.Module,
    train_loader,
    val_loader,
    opt: optim.Optimizer,
    loss_fn: callable,
    n_classes: int=6,
    epochs: int=1,
    early_stop: float=np.inf,
    fine_tune: bool=False,
    device: str="cpu",
    name: str="GiveMeName"):

    # Initialize variables
    train_acc_evo = []
    train_loss_evo = []
    val_acc_evo = []
    val_loss_evo = []
    best_loss = np.inf
    prev_loss = np.inf
    bad_epoch = 0
    if epochs<1:
        epochs=1

    os.makedirs(f"modelli/{name}", exist_ok=True)
    
    if fine_tune:
        for param in model.model.features.parameters():
            param.requires_grad = False
    
    # Load the model onto the device
    model.to(device)

    # Train the model for "epoch" number of epochs
    for epoch in range(epochs):
        model.train()
        predictions = []
        true = []
        #iterator = tqdm(train_loader)
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            # Make the model predict and do the backward step
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            
            predictions.append(y_pred)
            true.append(batch_y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            #print("Working fine")
            #iterator.set_description(f"Train loss: {loss.detach().cpu().numpy()}")
        
        # Compute loss and accuracy
        predictions = torch.cat(predictions, axis=0)
        true = torch.cat(true, axis=0)
        train_loss = loss_fn(predictions, true)
        predictions = torch.argmax(predictions, dim=1)
        train_acc = (predictions == true).float().mean()
        print(f"Epoch {epoch+1} --> Train loss: {train_loss} Train acc: {train_acc}")
        train_acc_evo.append(float(train_acc))
        train_loss_evo.append(float(train_loss))
        
        model.eval()
        with torch.no_grad():
            predictions = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Make the model predict and save those predictions
                y_pred = model(batch_x)
                predictions.append(y_pred)
                true.append(batch_y)
                
            # Compute loss and accuracy
            predictions = torch.cat(predictions, axis=0)
            true = torch.cat(true, axis=0)
            val_loss = loss_fn(predictions, true)
            predictions = torch.argmax(predictions, dim=1)
            val_acc = (predictions == true).float().mean()
            print(f"Epoch {epoch+1} --> Val loss: {val_loss} Val acc: {val_acc}\n")
            val_acc_evo.append(float(val_acc))
            val_loss_evo.append(float(val_loss))
        
        if val_loss < prev_loss:
            bad_epoch = 0
            prev_loss = val_loss
        else:
            bad_epoch += 1
            
        if bad_epoch>=early_stop:
            print(f"Stopping training at {epoch+1} epochs")
            break
            
        if val_loss < best_loss:
            # Save the best model
            torch.save(model.state_dict(), f"modelli/{name}/best_{name}.pt")
            print("New best model saved\n")
            best_loss = val_loss
            
    write_train_results(train_acc_evo=train_acc_evo, train_loss_evo=train_loss_evo, val_acc_evo=val_acc_evo, val_loss_evo=val_loss_evo, name=name, fine_tune=fine_tune)
    if fine_tune:
        for param in model.model.features.parameters():
            param.requires_grad = True
            
    return train_acc_evo, train_loss_evo, val_acc_evo, val_loss_evo


"Function to read the results of fine-tuning process"
def read_fine_results(folder: str):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    if folder is None:
        print("Insert a folder where to search.")
        return train_acc, train_loss, val_acc, val_loss
    try:
        with open(f"{folder}/train_loss_tune.txt", "r") as f:
            train_loss = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/train_acc_tune.txt", "r") as f:
            train_acc = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/val_loss_tune.txt", "r") as f:
            val_loss = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/val_acc_tune.txt", "r") as f:
            val_acc = [float(line.strip()) for line in f.readlines()]
            
        print(f"Fine-tuning accuracy and loss recovered.")
    except:
        print(f"In this folder ({folder}) there are no results.")
        
    return train_acc, train_loss, val_acc, val_loss

    
"Function to read the results of training process"
def read_train_results(folder: str):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    if folder is None:
        print("Insert a folder where to search.")
        return train_acc, train_loss, val_acc, val_loss
    try:
        with open(f"{folder}/train_loss.txt", "r") as f:
            train_loss = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/train_acc.txt", "r") as f:
            train_acc = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/val_loss.txt", "r") as f:
            val_loss = [float(line.strip()) for line in f.readlines()]
            
        with open(f"{folder}/val_acc.txt", "r") as f:
            val_acc = [float(line.strip()) for line in f.readlines()]
            
        print(f"Training accuracy and loss recovered.")
    except:
        print(f"In this folder ({folder}) there are no results.")
        
    return train_acc, train_loss, val_acc, val_loss
