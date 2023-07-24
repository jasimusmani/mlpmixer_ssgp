import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from src.components.data_ingestion import TrajectoryDataset
from src.models.mlp import MLP
from src.models.mlp_mixer import MlpMixer
from tqdm import tqdm
import matplotlib.pyplot as plt
import timeit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_name = "MLP_test1"
pos_train = np.load('/Users/jasim/mlpmixer_ssgp/dataset/position_wheel_train.npy')
force_train = np.load('/Users/jasim/mlpmixer_ssgp/dataset/force_wheel_train.npy')
pos_valid = np.load('/Users/jasim/mlpmixer_ssgp/dataset/position_wheel_valid.npy')
force_valid = np.load('/Users/jasim/mlpmixer_ssgp/dataset/force_wheel_valid.npy')

lr = 0.001
epochs = 100

dataset_train = TrajectoryDataset(pos_train, force_train)
dataset_valid = TrajectoryDataset(pos_valid, force_valid)
# create a DataLoader with batch size 32
batch_size = 1
train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(dataset_valid, batch_size, shuffle=False)
average_train_loss = []
train_loss = []
average_valid_loss = []
valid_loss = []

model1 = MLP()
model= MlpMixer(hidden_dim = 525, seq_len = 7, num_classes = 75, num_blocks = 1, pred_len = 1, tokens_mlp_dim = 7, channels_mlp_dim = 525)

def train(model):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for batch_idx, (train, gt) in enumerate(train_dataloader):
        train = train.reshape(40, 7, 25, 3)
        gt = gt.reshape(40, 1, 25, 3)
        running_loss = 0.0
        iteration = gt.shape[0]
        for i in tqdm(range(iteration)):
            optimizer.zero_grad()
            train_data = train[i,:,:,:]
            gt_data = gt[i, :, :, :]
            train_data=train_data.reshape(1,7,25,3)
            gt_data = gt_data.reshape(1,1, 25, 3)
            prediction = model(train_data)
            prediction=prediction.reshape(25,3)
            # print('prediction tensor shape',prediction.shape)
            loss = criterion(prediction, gt_data)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            print(f"Iteration Count: {batch_idx}, Training Loss: {loss.item()}")
            train_loss.append(loss.item())
            running_loss += loss.item()

        average_train_loss.append(running_loss/iteration)


    # model.eval()
    # with torch.no_grad():
    #     running_loss = 0
    #     for batch_idx, (valid, gt_valid) in enumerate(valid_dataloader):
    #         valid = valid.reshape(40, 7, 25, 3)
    #         gt_valid = gt_valid.reshape(40, 1, 25, 3)
    #         running_loss = 0.0
    #         iteration = gt_valid.shape[0]
    #         for i in tqdm(range(iteration)):
    #             valid_data = valid[i, :, :, :]
    #             gt_valid_data = gt_valid[i, :, :, :]
    #             valid_data = valid_data.reshape(1, 7, 25, 3)
    #             gt_valid_data = gt_valid_data.reshape(1, 1, 25, 3)
    #             prediction_valid = model(valid_data)
    #             loss = criterion(prediction_valid, gt_valid_data)
    #
    #             # print(f"Iteration Count: {batch_idx}, Training Loss: {loss.item()}")
    #             valid_loss.append(loss.item())
    #             running_loss += loss.item()
    #
    #         average_train_loss.append(running_loss / iteration)
    torch.save(model.state_dict(), f'model_{model_name}.pt')



# def rollout():


train(model)
plt.plot(train_loss)