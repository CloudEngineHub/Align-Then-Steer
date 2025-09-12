from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import torch
import bisect

class RDTDataset(Dataset):
    def __init__(self, path):
        datapath = path
        hdf5files = os.listdir(datapath)
        self.actions = []
        for hdf5file in hdf5files:
            hdf5path = os.path.join(datapath, hdf5file)
            h5filesource = h5py.File(hdf5path, 'r')
            trajectory = h5filesource['qpos'][:]
            for index in range(trajectory.shape[0]):
                self.actions.append(trajectory[index])
            h5filesource.close()
        self.actions = np.array(self.actions).astype(np.float32)
        self.len = self.actions.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return torch.tensor(self.actions[idx]) # dummy data to prevent breaking
    

class RDTSeqDataset(Dataset):
    def __init__(self, path, action_seq_length = 8):
        datapath = path
        hdf5files = os.listdir(datapath)
        self.actions = []
        self.idx_list = [0]
        self.len = 0
        self.action_seq_length = action_seq_length
        for hdf5file in hdf5files:
            hdf5path = os.path.join(datapath, hdf5file)
            h5filesource = h5py.File(hdf5path, 'r')
            action_each_file = h5filesource['qpos'][:]
            if action_each_file.shape[0] < action_seq_length:
                h5filesource.close()
                continue
            h5filesource.close()
            self.actions.append(np.array(action_each_file).astype(np.float32))
            self.len += len(action_each_file) - (action_seq_length - 1)
            self.idx_list.append(self.len)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        index = bisect.bisect_right(self.idx_list, idx) - 1
        start = max(0, idx-self.idx_list[index])
        return torch.tensor(self.actions[index][start:start+self.action_seq_length, :])