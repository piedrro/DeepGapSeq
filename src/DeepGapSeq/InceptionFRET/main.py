
import pickle
import os
import numpy as np
import importlib
import importlib.resources as resources
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import traceback
import json
from glob2 import glob
from DeepGapSeq.InceptionFRET.model import InceptionFRET
from DeepGapSeq.InceptionFRET.dataloader import load_dataset
from DeepGapSeq.InceptionFRET.trainer import Trainer
import sklearn








module_path = resources.files(importlib.import_module(f'DeepGapSeq.InceptionFRET'))
dataset_path = os.path.join(module_path, "deepgapseq_simulated_traces","dataset_2023_11_22.pkl")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)


# device
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    print("Training on CPU")
    device = torch.device('cpu')


ratio_train = 0.7
val_test_split = 0.5
BATCH_SIZE = 5
LEARNING_RATE = 0.01
EPOCHS = 100
channel_first = True


def calculate_accuracy(preds, labels):

    # Assuming preds and labels are of shape [batch_size, num_classes, sequence_length]
    # Convert probabilities to binary predictions
    preds_binary = preds.argmax(dim=1)  # Get the indices of the max values
    labels_binary = labels.argmax(dim=1)

    correct = (preds_binary == labels_binary).float()  # Convert to float for division
    accuracy = correct.mean()  # Calculate mean

    return accuracy.item()
    
    

if __name__ == '__main__':

    data_values = dataset['data'][:500]
    state_labels = dataset['labels'][:500]
    
    X_train, X_val, y_train, y_val = train_test_split(data_values, state_labels, train_size=ratio_train, random_state=42, shuffle=True)
    
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=val_test_split, random_state=42, shuffle=True)
    
    training_dataset = load_dataset(data=X_train, labels=y_train, augment=True, channel_first=channel_first)
    
    validation_dataset = load_dataset(data=X_val, labels=y_val, augment=False, channel_first=channel_first)
    
    test_dataset = load_dataset(data=X_test, labels=y_test, augment=False,channel_first=channel_first)
    
    trainloader = data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    valoader = data.DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    testloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = InceptionFRET(in_channels = 2, n_classes=2, depth=6).to(device)
    
    # for data, label in trainloader:
    #     plt.plot(data[0].T)
    #     print(data[0,0].max())
    #     print(data[0,1].max())
    #     # plt.plot(label[0].T)
    #     break
    #     pass
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        trainloader=trainloader,
        valoader=valoader,
        lr_scheduler=scheduler,
        tensorboard=True,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_dir=module_path,
        model_folder="models")
    
    trainer.train()
    
    
    
    








# training_dataset = load_dataset(data=data_values,
#                                 labels=state_labels,
#                                 augment=True,
#                                 num_classes=2)
#
# trainloader = torch.utils.data.DataLoader(
#     dataset=training_dataset,
#     batch_size=100,
#     shuffle=True)
#
#
# for data, label in trainloader:
#     print(data.shape)
#     print(label.shape)
#     print(label[0])
#     # trace = data[0,:,0]
#     # print(trace.min(), trace.max())
#
#     break
#
#
