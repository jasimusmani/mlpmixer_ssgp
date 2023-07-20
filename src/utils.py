import torch


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