import os
import torch
import sys
import time
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# a Dataset object
class LinesDataset(Dataset):
    def __init__(self, xdata, ydata):
        self.labels = ydata
        self.data = xdata

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        datum = self.data[idx]
        return datum, label


def createFiletree(modelname):
    # Parent Directories
    root = "."
    parent_dir = "outputs_from_models"
    path = os.path.join(root, parent_dir)
    path = os.path.join(path, modelname)
    Path(path).mkdir(parents=True, exist_ok=True)
    print("Saving model to '% s' " % modelname)

    # Leaf directories
    pathT = os.path.join(path, 'testing_data')
    Path(pathT).mkdir(parents=True, exist_ok=True)
    pathM = os.path.join(path, 'model')
    Path(pathM).mkdir(parents=True, exist_ok=True)
    pathP = os.path.join(path, 'plots')
    Path(pathP).mkdir(parents=True, exist_ok=True)

    return path, pathT, pathM, pathP


# Save a checkpoint
def checkpoint(model, pathM, filename):
    checkpath = os.path.join(pathM, filename)
    torch.save(model.state_dict(), checkpath)


# Resume training from a checkpoint
def resume(model, pathM, filename):
    checkpath = os.path.join(pathM, filename)
    model.load_state_dict(torch.load(checkpath))
