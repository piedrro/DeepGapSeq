import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from helper import rescale01

def preprocess_data(x):

    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(x).reshape(-1,1))
    x = np.squeeze(x)

    # for i in range(len(x)):
    #     if -np.std(x) < x[i] < np.std(x):
    #         x[i] = 0

    return x


def split_list(data, chunk_size = 200):

    split_data = [] 
    
    for dat in data:
        
        dat_split = np.split(dat,range(0, len(dat), 200), axis=0)
        
        dat_split = [list(x) for x in dat_split if len(x) == 200]
        
        split_data.extend(dat_split)
        
    return split_data


def read_gapseq_data(file_paths, label, trace_limit = 1200):
    
    data_x = []
    file_names = []

    for file_path in file_paths:
        with open(file_path) as f:
            
            d = json.load(f)
            data = np.array(d["data"])
            data = [dat for dat in data]
            
            for sequence in data:
                if len(sequence) > 200:
                    
                    sequence = sequence[:trace_limit]
                    sequence = preprocess_data(sequence)
                    
                    data_x.append(list(sequence))
                    file_names.append(os.path.basename(file_path))

    return data_x, [label] * len(data_x), file_names


def shuffle_train_data(train_data):
      
    dict_names = list(train_data.keys())     
    dict_values = list(zip(*[value for key,value in train_data.items()]))
    
    random.shuffle(dict_values)
    
    dict_values = list(zip(*dict_values))
    
    train_data = {key:list(dict_values[index]) for index,key in enumerate(train_data.keys())}
    
    return train_data
                    

def limit_train_data(train_data, num_files):
    
    for key,value in train_data.items():
        
        train_data[key] = value[:num_files]
        
    return train_data


def split_dataset(X,y,file_names,ratio_train,val_test_split):
    

    dataset = {"X":np.array(X),"y":np.array(y),"file_names":np.array(file_names)}
    
    train_dataset = {"X":[],"y":[],"file_names":[]}
    validation_dataset = {"X":[],"y":[],"file_names":[]}
    test_dataset = {"X":[],"y":[],"file_names":[]}
    
    for label in np.unique(dataset["y"]):
        
        label_file_names = np.unique(np.extract(dataset["y"]==label,dataset["file_names"]))
        
        for file_name in label_file_names:
            
            indices = np.argwhere(dataset["file_names"]==file_name).flatten()
            
            
            
            train_indices, val_indices = train_test_split(indices,
                                                          train_size=ratio_train,
                                                          shuffle=True)
            
            val_indices, test_indices = train_test_split(val_indices,
                                                          train_size=val_test_split,
                                                          shuffle=True)
            
            for key,value in dataset.items():
                
                train_data = dataset[key][train_indices].tolist()
                validation_data = dataset[key][val_indices].tolist()
                test_data = dataset[key][test_indices].tolist()
                
                train_dataset[key].extend(train_data)
                validation_dataset[key].extend(validation_data)
                test_dataset[key].extend(test_data)
            
    train_dataset = shuffle_train_data(train_dataset) 
    validation_dataset = shuffle_train_data(validation_dataset) 
    test_dataset = shuffle_train_data(test_dataset)

    return train_dataset, validation_dataset, test_dataset