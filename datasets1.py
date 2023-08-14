import glob
import os

import cupy as cp
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, mode,datapath):
        self.path = datapath
        self.mode = mode
        self.files = sorted(glob.glob(os.path.join(self.path, self.mode) + "/*.mat"))
    
    def __getitem__(self, index: int):
        file = sio.loadmat(self.files[index])
        events = file['events']     #[nevents, 6]
        image = file['image']       #[128, 128]

        return {'events': events, "image": image}
    
    def __len__(self):
        return len(self.files)     
    

