from torch.utils import data
import torch
import torch.nn.functional as F
import sklearn
import numpy as np
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse
from sklearn.preprocessing import minmax_scale
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse


class load_dataset(data.Dataset):

    def __init__(self, data=[], labels=[], num_classes=2, augment=None,channel_first=True):
        self.augment = augment
        self.data = data
        self.labels = labels
        self.num_classes = num_classes
        self.channel_first = channel_first

    def __len__(self):
        return len(self.data)

    def augment_traces(self, X):

        augmenter = (TimeWarp(n_speed_change=5, max_speed_ratio=3) @ 0.5 + Quantize(n_levels=[20, 30, 50, 100]) @ 0.5 + Drift(max_drift=(0.01, 0.1), n_drift_points=5) @ 0.5 + Reverse() @ 0.5)

        X = augmenter.augment(np.array(X))

        return X

    def min_max_normalize(self, x):
        
         min_val = x.min(dim=0, keepdim=True)[0]
         max_val = x.max(dim=0, keepdim=True)[0]
         x_normalized = (x - min_val) / (max_val - min_val)
         
         return x_normalized

    def postprocess(self, X, y):
        

        # Typecasting
        X = torch.from_numpy(X.copy()).float()
        
        X = self.min_max_normalize(X)
        
        y = torch.tensor(y, dtype=torch.long)
        y = F.one_hot(y, num_classes=self.num_classes).float()
        
        if self.channel_first:
            X = X.t() # Swap the channel and sequence length dimensions
            y = y.t()

        return X, y

    def __getitem__(self, index: int):
        X, y = self.data[index], self.labels[index]

        if self.augment:
            X = self.augment_traces(X)

        X, y = self.postprocess(X, y)

        return X, y