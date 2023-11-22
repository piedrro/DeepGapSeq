
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

from DeepGapSeq.InceptionFRET.model import InceptionFRET
from DeepGapSeq.InceptionFRET.dataloader import load_dataset
from DeepGapSeq.InceptionFRET.trainer import Trainer



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
BATCH_SIZE = 3
LEARNING_RATE = 0.5
EPOCHS = 100


if __name__ == '__main__':

    X = dataset['data']
    y = dataset['labels']

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=ratio_train, random_state=42, shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, train_size=val_test_split, random_state=42, shuffle=True)

    training_dataset = load_dataset(data=X_train, labels=y_train, augment=True)

    validation_dataset = load_dataset(data=X_val, labels=y_val, augment=False)

    test_dataset = load_dataset(data=X_test, labels=y_test, augment=False)

    trainloader = data.DataLoader(dataset=training_dataset, batch_size=BATCH_SIZE, shuffle=True)

    valoader = data.DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    testloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = InceptionFRET(in_channels = 2, n_classes=2).to(device)

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
