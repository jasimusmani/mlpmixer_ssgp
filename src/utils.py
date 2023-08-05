import torch
import numpy as np

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    # Calculate the velocity sequence from the position sequence.
    velocity_sequence = position_sequence[:, 1:] - position_sequence[:, :-1]

    # We want the noise scale in the velocity at the last step to be fixed.
    num_velocities = velocity_sequence.size(1)
    velocity_sequence_noise = torch.normal(mean=0.0, std=noise_std_last_step / num_velocities ** 0.5,
                                           size=velocity_sequence.size(), dtype=position_sequence.dtype)

    # Apply the random walk to the velocity sequence.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate the noise in the velocity to the positions.
    position_sequence_noise = torch.cat([torch.zeros_like(velocity_sequence_noise[:, :1]),
                                         torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    # Add the position_sequence_noise to the original position_sequence to get the noisy positions.
    noisy_position_sequence = position_sequence + position_sequence_noise

    return noisy_position_sequence


def mpjpe_error(batch_pred, batch_gt):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_pred, 2, 1))


def euclidean_transform_3D(A, B):
    '''
        A,B - Nx3 matrix
        return:
            R - 3x3 rotation matrix
            t = 3x1 column vector
    '''
    assert len(A) == len(B)

    N = A.shape[0];

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)


    centroid_A = centroid_A.reshape(1, 3)
    centroid_B = centroid_B.reshape(1, 3)

    # centre matrices
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(BB), AA)
    U, S, Vt = np.linalg.svd(H)

    # resulting rotation
    asa = np.linalg.det(U @ Vt)
    diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, asa]])
    R = U @ diag @ Vt
    # handle svd sign problem
    if np.linalg.det(R) < 0:
        #        print ("sign")
        Vt[2, :] *= -1
        #         R = Vt.T * U.T
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    next_position = (np.matmul(R, np.transpose(A))) + np.tile(t, (1, 16))
    next_position = np.transpose(next_position)
    next_position = next_position.reshape(16, 3)

    return next_position