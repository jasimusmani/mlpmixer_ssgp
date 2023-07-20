import torch
import numpy as np
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, pos_data_file,force_data_file):
        self.pos_data = pos_data_file
        self.force_data = force_data_file
        self.num_frames = self.pos_data.shape[0]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):

        input_data_position = self.pos_data[idx, :, :]
        input_data_force = self.force_data[idx, :, :]
        input_data_position = input_data_position.reshape(321,24,3)
        input_data_force = input_data_force.reshape(321,1,3)
        input_tensor = np.concatenate((input_data_position, input_data_force), axis=1)
        tr_i = input_tensor[0:320,:,:]
        for i in range(218):
            tr = tr_i.reshape(40,8,25,3)
            tr_final = tr[:,0:7,:,:]
            gt_final = tr[:,7:,:,:]
        # Convert to PyTorch tensors and return
        train = torch.from_numpy(tr_final).float()
        gt = torch.from_numpy(gt_final).float()

        return train, gt