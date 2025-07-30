import numpy as np
#import os
import torch
import torch.nn as nn
#import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

"""Function to test a NN model"""    
def test_this_model(
    model: nn.Module,
    test_loader,
    loss_fn: callable,
    iter: int=0,
    n_classes: int=6,
    report: bool=False,
    device: str="cpu",
    name: str="GiveMeName"):
    
    c=0
    model.to(device)
    # If iter is not set, test on whole test dataset
    if iter<1:
        iter = np.inf
        print("Starting test...")
    else:
        print(f"Testing only on {iter} test batches...")
        
    predictions = []
    true = []
    test_acc = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Exit loop if number of desired batch is reached
            if c>=iter:
                break
            batch_x = batch_x.to(device)
            # Convert target to one-hot encoding
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)
            predictions.append(y_pred)
            true.append(batch_y)
            c+=1
            
        predictions = torch.cat(predictions, axis=0)
        true = torch.cat(true, axis=0)
        test_loss = loss_fn(predictions, true)
        predictions = torch.argmax(predictions, dim=1)
        test_acc = (predictions == true).float().mean()
        predictions = predictions.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        print(f"Test loss: {test_loss} Test accuracy: {test_acc}\n")
        with open(f"modelli/{name}/test_result.txt", "w") as f:
            f.write(f"{test_acc}\n{test_loss}")
    
    # If report is set to True, print confusion matrix and report
    if report:
        class_names = list(test_loader.dataset.class_to_idx.values())
        report = classification_report(true, predictions, target_names=class_names)
        print(report)
        matrix = confusion_matrix(true, predictions)
        print(matrix)
        with open(f"modelli/{name}/test_result.txt", "w") as f:
            f.write(f"{test_acc}\n{test_loss}\n")
            f.write(f"\n{report}\n")
            f.write(f"\n{matrix}")

    return predictions, true

"Function to import test results of a model"
def read_test_results(folder: str=None, show: bool=True):
    test_acc = 0
    test_loss = 0
    if folder is None:
        print("Insert a folder where to search.")
        return test_acc, test_loss
    try:
        with open(f"{folder}/test_results.txt", "r") as f:
            test_acc = float(f.readlines()[0].strip())
            test_loss = float(f.readlines()[1].strip())
            if show:
                print(f"Test accuracy: {test_acc}")
                print(f"Test loss: {test_loss}")
            else:
                print(f"Test accuracy and loss recovered.")
    except:
        print(f"In this folder ({folder}) there are no results.")
    
    return test_acc, test_loss
