import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.components.data_ingestion import TrajectoryDataset
from src.models.mlp import MLP
from src.models.mlp_mixer import MlpMixer
from tqdm import tqdm
import pickle
import os
import timeit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "MLP_test1"
pos_train = np.load('/Users/jasim/mlpmixer_ssgp/dataset/position_wheel_train.npy')
force_train = np.load('/Users/jasim/mlpmixer_ssgp/dataset/force_wheel_train.npy')
pos_valid = np.load('/Users/jasim/mlpmixer_ssgp/dataset/position_wheel_valid.npy')
force_valid = np.load('/Users/jasim/mlpmixer_ssgp/dataset/force_wheel_valid.npy')
pos_test = np.load('/Users/jasim/mlpmixer_ssgp/dataset/position_wheel_test.npy')
force_test = np.load('/Users/jasim/mlpmixer_ssgp/dataset/force_wheel_test.npy')

output_folder = f'/Users/jasim/mlpmixer_ssgp/rollout/{model_name}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model_save = f'/Users/jasim/mlpmixer_ssgp/saved_model/{model_name}/{model_name}.pt'

log_dir = f'/Users/jasim/mlpmixer_ssgp/saved_model/{model_name}'
writer = SummaryWriter(log_dir=log_dir)

lr = 0.001
num_epochs = 100
batch_size = 1

dataset_train = TrajectoryDataset(pos_train, force_train)
dataset_valid = TrajectoryDataset(pos_valid, force_valid)
dataset_test =  TrajectoryDataset(pos_test, force_test)

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

average_train_loss = []
train_loss = []
average_valid_loss = []
valid_loss = []

model1 = MLP()
model= MlpMixer(hidden_dim = 525, seq_len = 7, num_classes = 75, num_blocks = 1, pred_len = 1, tokens_mlp_dim = 7, channels_mlp_dim = 525)

def train(model):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        progress_bar = tqdm(total=num_epochs, desc=f' Model Training: Epoch {epoch}', position=0,
                            leave=True)
        for batch_idx, (train, gt) in enumerate(train_dataloader):
            train = train.reshape(40, 7, 25, 3)
            gt = gt.reshape(40, 1, 25, 3)
            running_loss = 0.0
            iteration = gt.shape[0]
            progress_bar = tqdm(total=iteration, desc=f' Model Training: Epoch {epoch} Batch {batch_idx}', position=0,
                                leave=True)

            for i in range(iteration):
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
                # print(f"Iteration Count: {batch_idx}, Training Loss: {loss.item()}")
                train_loss.append(loss.item())
                running_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), batch_idx * len(train_dataloader) + i)
                progress_bar.update(1)
                progress_bar.set_postfix({'Loss': running_loss / (i + 1)})

            progress_bar.close()

            model.eval()
            with torch.no_grad():
                running_loss = 0
                for batch_idx, (valid, gt_valid) in enumerate(valid_dataloader):
                    valid = valid.reshape(40, 7, 25, 3)
                    gt_valid = gt_valid.reshape(40, 1, 25, 3)
                    running_loss = 0.0
                    iteration = gt_valid.shape[0]

                    # progress_bar = tqdm(total=iteration, desc=f' Model Validation: Batch {batch_idx}', position=0,
                    #                     leave=True)

                    for i in range(iteration):
                        valid_data = valid[i, :, :, :]
                        gt_valid_data = gt_valid[i, :, :, :]
                        valid_data = valid_data.reshape(1, 7, 25, 3)
                        gt_valid_data = gt_valid_data.reshape(1, 1, 25, 3)
                        prediction_valid = model(valid_data)
                        prediction_valid = prediction_valid.reshape(25, 3)

                        loss = criterion(prediction_valid, gt_valid_data)

                        # print(f"Iteration Count: {batch_idx}, Validation Loss: {loss.item()}")
                        valid_loss.append(loss.item())
                        running_loss += loss.item()
                    #     progress_bar.update(1)
                    #     progress_bar.set_postfix({'Loss': running_loss / (i + 1)})
                    #
                    # progress_bar.close()

                    average_train_loss.append(running_loss / iteration)
        progress_bar.update(1)
        progress_bar.set_postfix({'Loss': running_loss / (i + 1)})
    progress_bar.close()
    writer.close()
    torch.save(model.state_dict(), model_save)



def rollout(model, output_folder):

    model.load_state_dict(torch.load(model_save))
    model.eval()
    with torch.no_grad():  # Disable gradient calculation during testing
        for batch_idx, (test, _ ) in enumerate(test_dataloader):
            rollout_prediction = []
            test = test.reshape(40, 7, 25, 3)
            # gt_test = gt_test.reshape(40, 1, 25, 3)

            # running_loss = 0.0
            # iteration = gt_test.shape[0]

            initial_data = test[0]
            initial_data_in = initial_data.reshape(1, 7, 25, 3)
            start = timeit.default_timer()
            for i in range(314):

                prediction_test = model(initial_data_in)
                prediction_test = prediction_test.reshape(1,25,3)
                rollout_prediction.append(prediction_test)
                initial_data = initial_data_in[:,1:,:,:]
                prediction_test = prediction_test.reshape(1, 1, 25, 3)
                initial_data = torch.cat([initial_data,prediction_test], dim=1)
            stop = timeit.default_timer()
            print('Time (with serializing output): ', stop - start)
            final_rollout = torch.stack(rollout_prediction)
            final_rollout = final_rollout.reshape(314, 25, 3)
            initial_positions = initial_data.reshape(7, 25, 3)
            initial_positions = initial_positions[:,0:24,:]
            initial_forces = initial_positions[:,24:,:]
            predicted_rollout = final_rollout[:,0:24,:]
            predicted_forces = predicted_rollout[:,24:,:]
            ground_truth_rollout = pos_test[batch_idx,7:,:]
            ground_truth_rollout = ground_truth_rollout.reshape(-1,24,3)
            ground_truth_forces = force_test[batch_idx,7:,:]


            model_rollout = {
                'initial_positions' : initial_positions.numpy(),
                'predicted_rollout' : predicted_rollout.numpy(),
                'ground_truth_rollout' : ground_truth_rollout,
                'initial_forces' : initial_forces.numpy(),
                'predicted_forces' : predicted_forces.numpy(),
                'ground_truth_forces' : ground_truth_forces,
            }
            save_file_path = os.path.join(output_folder, f'model_rollout_{batch_idx}.pkl')
            with open(save_file_path, 'wb') as f:
                pickle.dump(model_rollout, f)
    return final_rollout




train(model)
# rollout(model, output_folder)
