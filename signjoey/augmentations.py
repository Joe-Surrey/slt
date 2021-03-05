import torch
import numpy as np



def centre_and_scale(batch):
    """
    Centre by shoulders then scale by distance between shoulders and hips
    """
    x_indexes = np.array(range(0, batch.shape[-1], 3))
    y_indexes = np.array(range(1, batch.shape[-1], 3))

    # centre by centre of shoulders
    shoulder_x_means = np.array([(batch[:, 5 * 3] + batch[:, 6 * 3]) / 2])
    shoulder_y_means = np.array([(batch[:, (5 * 3) + 1] + batch[:, (6 * 3) + 1]) / 2])

    batch[:, x_indexes] -= shoulder_x_means.T
    batch[:, y_indexes] -= shoulder_y_means.T

    # Scale by distance between shoulder centre and hip centre
    hip_x_means = np.array([(batch[:, (11 * 3)] + batch[:, (12 * 3)]) / 2])
    hip_y_means = np.array([(batch[:, (11 * 3) + 1] + batch[:, (12 * 3) + 1]) / 2])

    hip = np.concatenate([hip_x_means, hip_y_means])
    shoulder = np.concatenate([shoulder_x_means, shoulder_y_means])

    scale_factors = dist = np.linalg.norm(hip-shoulder)

    batch[:, x_indexes] /= scale_factors.T
    batch[:, y_indexes] /= scale_factors.T
    return batch


def to_tensor(batch):
    """
    Convert np array to tensor
    """
    batch = torch.Tensor(batch)
    return batch

def remove_lower_body(batch):
    """
    Remove the lower body (keypoints 13 - 24 inc.)
    Note this affects the indexes of all subsequent keypoints
    """
    batch = np.delete(batch, (range(13 * 3, 25 * 3)), axis=1)
    return batch

def load_augment(batch):
    """
    Augmentations to do to a batch when loading
    This is where removing keypoints etc. should go
    The input will be a numpy array and output a torch Tensor
    """
    batch = remove_lower_body(batch)
    batch = centre_and_scale(batch)
    batch = to_tensor(batch)

    return batch

def train_augment(batch):
    """
    Augmentations to do when training
    This is where random flips etc. should go
    The input and output will be a torch Tensor
    """
    return batch