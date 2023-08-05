import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.components.data_ingestion import TrajectoryDataset, ValidationTrajectoryDataset
from src.utils import mpjpe_error, euclidean_transform_3D
from src.models.mlp import MLP
from src.models.mlp_mixer import MlpMixer
from tqdm import tqdm
import pickle
import os
import timeit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_name = "MLP_mixer_8k_981_mpje"
pos_train = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/pos_train_981.npy')
force_train = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/force_train_981.npy')
pos_valid = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/pos_valid_981.npy')
force_valid = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/force_valid_981.npy')
pos_test = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/pos_test_981.npy')
force_test = np.load('/home/jasimusmani/mlpmixer_ssgp-main/dataset/force_test_981.npy')

output_folder = f'/home/jasimusmani/mlpmixer_ssgp-main/rollout/{model_name}'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

model_save = f'/home/jasimusmani/mlpmixer_ssgp-main/{model_name}/{model_name}.pt'

log_dir = f'/home/jasimusmani/mlpmixer_ssgp-main/{model_name}'
writer = SummaryWriter(log_dir=log_dir)

lr = 0.0001
num_epochs = 8000
batch_size = 1

dataset_train = TrajectoryDataset(pos_train, force_train)
dataset_valid = TrajectoryDataset(pos_valid, force_valid)
dataset_valid_gt = ValidationTrajectoryDataset(pos_valid, force_valid)
dataset_test = TrajectoryDataset(pos_test, force_test)

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

average_train_loss = []
train_loss = []
average_valid_loss = []
valid_loss = []

# model1 = MLP()
model = MlpMixer(hidden_dim=525, seq_len=7, num_classes=75, num_blocks=4, pred_len=1, tokens_mlp_dim=7,
                 channels_mlp_dim=525)
model = model.to(device)


def train(model):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-04)
    model.train()

    checkpoint_dir = f'/home/jasimusmani/mlpmixer_ssgp-main/{model_name}/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Load checkpoint if available
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
    else:
        start_epoch = 0

    best_valid_loss = float('inf')  # Initialize the best validation loss

    for epoch in range(start_epoch, num_epochs):
        progress_bar_main = tqdm(total=num_epochs, desc=f' Model Training: Epoch {epoch}', position=0,
                            leave=True)
        for batch_idx, (train, gt) in enumerate(train_dataloader):
            train = train.to(device)
            gt = gt.to(device)
            train = train.reshape(40, 7, 25, 3)
            gt = gt.reshape(40, 1, 25, 3)
            running_loss = 0.0
            iteration = gt.shape[0]
            progress_bar = tqdm(total=iteration, desc=f' Model Training: Epoch {epoch} Batch {batch_idx}', position=0,
                                leave=True)

            for i in range(iteration):
                optimizer.zero_grad()
                train_data = train[i, :, :, :]
                gt_data = gt[i, :, :, :]
                train_data = train_data.reshape(1, 7, 25, 3)
                gt_data = gt_data.reshape(1, 1, 25, 3)
                train_data = train_data.to(device)
                gt_data = gt_data.to(device)
                prediction = model(train_data)
                prediction = prediction.reshape(25, 3)
                # print('prediction tensor shape',prediction.shape)
                # loss1 = criterion(prediction, gt_data)
                loss = mpjpe_error(prediction, gt_data)
                # loss = loss1+loss2

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
                valid_running_loss = 0
                for batch_idx_valid, (valid, gt_valid) in enumerate(valid_dataloader):
                    iteration_valid = gt_valid.shape[0]
                    progress_bar_valid = tqdm(total=iteration_valid, desc=f' Model Validation: Batch {batch_idx_valid}',
                                        position=0,
                                        leave=True)
                    valid = valid.to(device)
                    valid = valid.reshape(40, 7, 25, 3)
                    running_loss = 0.0

                    valid_gt = dataset_valid_gt[batch_idx_valid]
                    valid_gt = valid_gt.reshape(-1,25,3)

                    valid_prediction = []
                    valid_gt = valid_gt.to(device)


                    initial_valid = valid[0]
                    initial_valid_data = initial_valid.reshape(1, 7, 25, 3)

                    initial_valid_data = initial_valid_data.to(device)
                    for i in range(314):

                        prediction_valid = model(initial_valid_data)
                        prediction_valid = prediction_valid.reshape(1,25, 3)

                        valid_prediction.append(prediction_valid)
                        initial_valid_data = initial_valid_data[:, 1:, :, :]
                        prediction_valid = prediction_valid.reshape(1, 1, 25, 3)
                        initial_valid_data = torch.cat([initial_valid_data, prediction_valid], dim=1)

                    final_valid = torch.stack(valid_prediction)
                    final_valid = final_valid.reshape(314, 25, 3)
                    final_valid = torch.cat([initial_valid, final_valid], dim = 0)
                    loss = mpjpe_error(final_valid, valid_gt)
                        # print(f"Iteration Count: {batch_idx}, Validation Loss: {loss.item()}")
                    valid_loss.append(loss.item())
                    valid_running_loss += loss.item()
                    progress_bar_valid.update(1)
                    progress_bar_valid.set_postfix({'Loss': valid_running_loss / (batch_idx_valid + 1)})
                average_valid_loss.append(valid_running_loss / len(valid_dataloader))
                progress_bar_valid.close()


                # Check if the current validation loss is the best so far
                if average_valid_loss[-1] < best_valid_loss:
                    best_valid_loss = average_valid_loss[-1]

                    # Save the model checkpoint whenever the validation loss improves
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_file)

            progress_bar_main.update(1)
            progress_bar_main.set_postfix({'Loss': running_loss / (epoch + 1)})
        progress_bar_main.close()
    writer.close()
    torch.save(model.state_dict(), model_save)


def rollout(model, output_folder):
    model.load_state_dict(torch.load(model_save))
    model.eval()
    with torch.no_grad():  # Disable gradient calculation during testing
        for batch_idx, (test, _) in enumerate(test_dataloader):
            rollout_prediction = []
            test = test.reshape(40, 7, 25, 3)
            test = test.to(device)
            # gt_test = gt_test.reshape(40, 1, 25, 3)

            # running_loss = 0.0
            # iteration = gt_test.shape[0]

            initial_data = test[0]
            initial_data = initial_data.reshape(1, 7, 25, 3)
            start = timeit.default_timer()
            initial_data = initial_data.to(device)
            for i in range(314):
                prediction_test = model(initial_data)
                prediction_test = prediction_test.reshape(1, 25, 3)
                A = initial_data[:, 6:, :16, :].numpy()
                A = A.reshape(16, 3)
                B = prediction_test[:, 0:16, :].numpy()
                B = B.reshape(16, 3)
                corrected_position = euclidean_transform_3D(A, B)
                pca_pred = prediction_test[:, 16:, :].numpy()
                pca_pred = pca_pred.reshape(9, 3)
                corrected_pred = np.concatenate((corrected_position, pca_pred), axis=0)
                # rollout_prediction.append(prediction_test)
                corrected_pred = torch.from_numpy(corrected_pred.reshape(1, 25, 3))
                rollout_prediction.append(corrected_pred)
                initial_data = initial_data[:, 1:, :, :]
                prediction_test = prediction_test.reshape(1, 1, 25, 3)
                initial_data = torch.cat([initial_data, prediction_test], dim=1)
            stop = timeit.default_timer()
            print('Time (with serializing output): ', stop - start)
            final_rollout = torch.stack(rollout_prediction)
            final_rollout = final_rollout.reshape(314, 25, 3)
            initial_input = test[0]
            initial_positions = initial_input.reshape(7, 25, 3)
            initial_forces = initial_positions[:, 24:, :]
            initial_forces = initial_forces.reshape(7, 3)
            initial_positions = initial_positions[:, 0:24, :]
            predicted_rollout = final_rollout[:, 0:24, :]
            predicted_forces = final_rollout[:, 24:, :]
            ground_truth_rollout = pos_test[batch_idx, 7:, :]
            ground_truth_rollout = ground_truth_rollout.reshape(-1, 24, 3)
            ground_truth_forces = force_test[batch_idx, 7:, :]
            ground_truth_forces = ground_truth_forces.reshape(-1, 3)
            particle_types = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6])
            print(initial_forces.shape)
            model_rollout = {
                'initial_positions': initial_positions.cpu().numpy(),
                'predicted_rollout': predicted_rollout.cpu().numpy(),
                'ground_truth_rollout': ground_truth_rollout,
                'initial_forces': initial_forces.cpu().numpy(),
                'predicted_forces': predicted_forces.cpu().numpy(),
                'ground_truth_forces': ground_truth_forces,
                'particle_types': particle_types
            }
            save_file_path = os.path.join(output_folder, f'model_rollout_{batch_idx}.pkl')
            with open(save_file_path, 'wb') as f:
                pickle.dump(model_rollout, f)
    return final_rollout


train(model)
rollout(model, output_folder)
