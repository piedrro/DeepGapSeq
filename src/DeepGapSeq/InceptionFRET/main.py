
import pickle
import os
import numpy as np
import importlib
import importlib.resources as resources
from DeepGapSeq.InceptionFRET.model import InceptionFRET
from DeepGapSeq.InceptionFRET.dataloader import load_dataset
import torch


dataset_directory = resources.files(importlib.import_module(f'DeepGapSeq.InceptionFRET'))
dataset_path = os.path.join(dataset_directory, "deepgapseq_simulated_traces","dataset_2023_11_22.pkl")

with open(dataset_path, 'rb') as f:
    dataset = pickle.load(f)

data_values = dataset['data']
state_labels = dataset['labels']

training_dataset = load_dataset(data=data_values,
                                labels=state_labels,
                                augment=True,
                                num_classes=2)

trainloader = torch.utils.data.DataLoader(
    dataset=training_dataset,
    batch_size=100,
    shuffle=True)



for data, label in trainloader:
    print(data.shape)
    print(label.shape)
    print(label[0])
    # trace = data[0,:,0]
    # print(trace.min(), trace.max())

    break


