import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np


class CTDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, _ = pkload(path)
        y = pkload('D:/DATA/Duke/XCAT/phan.pkl')
        y = np.flip(y, 1)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

class CycDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

class CTSegDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x, x_seg = pkload(self.paths[index])
        y = pkload('D:/DATA/Duke/XCAT/phan.pkl')
        y_seg = pkload('D:/DATA/Duke/XCAT/label.pkl')
        y = np.flip(y, 1)
        y_seg = np.flip(y_seg, 1)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x,y = self.transforms([x, y])
        x_seg, y_seg = self.transforms([x_seg, y_seg])
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)

        x_seg = np.ascontiguousarray(x_seg).astype(np.uint8)
        x_seg = torch.from_numpy(x_seg)
        y_seg = np.ascontiguousarray(y_seg).astype(np.uint8)
        y_seg = torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)