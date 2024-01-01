import tensorflow as tf
import numpy as np
from functools import partial
from tsaug import  Crop, Quantize, Drift, Reverse, AddNoise, Convolve, Drift, Dropout, Pool, Resize
import json
from sklearn.preprocessing import StandardScaler

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

def preprocess_data(x):

    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(x).reshape(-1,1))
    x = np.squeeze(x)

    # for i in range(len(x)):
    #     if -np.std(x) < x[i] < np.std(x):
    #         x[i] = 0
    return x

def augment_traces(X):
        
        augmenter = (
                    # TimeWarp(n_speed_change=5, max_speed_ratio=3)@ 0.5 +
                    Quantize(n_levels=[20, 30, 50, 100]) @ 0.5 + 
                    Drift(max_drift=(0.01, 0.1), n_drift_points=5) @ 0.5 + 
                    Reverse() @ 0.5 + 
                    AddNoise(scale=0.1) @ 0.15 +
                    # Convolve(window="flattop", size=11)  @ 0.1 + 
                    Drift(max_drift=0.2, n_drift_points=5) @ 0.15 +
                    Dropout(p=0.1, size=(1,2), fill=float(0), per_channel=True) @ 0.15
                    )
        # Rasched - add more augmentations
        # Rasched - I noticed TimeWarp made the accuracy lower - this makes sense if it changes the duration of spikes which is critical in 
        # distinguising the two classes
        # X_aug = AddNoise(scale=0.1).augment(np.array(X[a])) can make it 0.05 for smaller effect
        # X_aug = Convolve(window="flattop", size=11).augment(np.array(X[a])) 
        # X_aug = Crop(size=300).augment(np.array(X[a]))  use this but make it custom so it crops a random part out and sets it to 0 ??
        # X_aug = Drift(max_drift=0.2, n_drift_points=5).augment(np.array(X[a]))
        # X_aug = Dropout(p=0.1, size=(1,2), fill=float(0), per_channel=True).augment(np.array(X[a]))
        # X_aug = Pool(size=2).augment(np.array(X[a])) not sure about this one

        X = augmenter.augment(np.array(X))
        
        return X

def postprocess(self, X, y, num_classes=2):
            
    # Typecasting
    X = np.array(X)
    
    X = torch.from_numpy(X.copy()).float()
    X = torch.unsqueeze(X,0)
    y = F.one_hot(torch.tensor(y), num_classes).float()

    return X, y

def get_data_labels(file_path, label, num_classes=2, trace_limit=1200, augment=True):
    
    X, y, _ = read_gapseq_data(file_paths, label, trace_limit)
    X = preprocess_data(X)
    if augment:
        X = augment_traces(X)
    X, y = postprocess(X, y, num_classes)

    return X, y

