import torch
import numpy as np
import random


def centre_and_scale(example):
    """
    Centre by shoulders then scale by distance between shoulders and hips
    """
    x_indexes = np.array(range(0, example.shape[-1], 3))
    y_indexes = np.array(range(1, example.shape[-1], 3))

    # centre by centre of shoulders
    shoulder_x_means = np.array([(example[:, 5 * 3] + example[:, 6 * 3]) / 2])
    shoulder_y_means = np.array([(example[:, (5 * 3) + 1] + example[:, (6 * 3) + 1]) / 2])

    example[:, x_indexes] -= shoulder_x_means.T
    example[:, y_indexes] -= shoulder_y_means.T

    # Scale by distance between shoulder centre and hip centre
    #hip_x_means = np.array([(example[:, (11 * 3)] + example[:, (12 * 3)]) / 2])
    #hip_y_means = np.array([(example[:, (11 * 3) + 1] + example[:, (12 * 3) + 1]) / 2])

    #hip = np.concatenate([hip_x_means, hip_y_means])
    #shoulder = np.concatenate([shoulder_x_means, shoulder_y_means])

    #scale_factors = dist = np.linalg.norm(hip-shoulder)

    #example[:, x_indexes] /= scale_factors.T
    #example[:, y_indexes] /= scale_factors.T

    return example


def to_tensor(data):
    """
    Convert np array to tensor
    """
    data = torch.Tensor(data)

    return data


def to_np(batch):
    batch = batch.numpy()

    return batch


def remove_lower_body(example):
    """
    Remove the lower body (keypoints 13 - 24 inc.)
    Note this affects the indexes of all subsequent keypoints
    """
    example = np.delete(example, (range(13 * 3, 25 * 3)), axis=1)

    return example


def load_augment(batch):
    """
    Augmentations to do to when loading
    This is where removing keypoints etc. should go
    The input will be a numpy array and output a torch Tensor
    The augmentations are applied to individual examples
    """
    batch = remove_lower_body(batch)
    batch = centre_and_scale(batch)
    batch = to_tensor(batch)

    return batch


def random_flip(batch):
    x_indexes = np.array(range(0, batch.shape[-1], 3))
    if random.random() > 0.5:
        batch[:, :, x_indexes] = -batch[:, :, x_indexes]
    return batch


def random_rotation(batch):
    x_indexes = np.array(range(0, batch.shape[-1], 3))
    y_indexes = np.array(range(1, batch.shape[-1], 3))

    theta = (random.random() / 10) - 0.05  # (range 0-0.1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # x = xcos(o) - ysin(o)
    batch[:, :, x_indexes] = (batch[:, :, x_indexes] * cos_theta) - (batch[:, :, y_indexes] * sin_theta)
    # y = xsin(o) + y cos(o)
    batch[:, :, y_indexes] = (batch[:, :, x_indexes] * sin_theta) + (batch[:, :, y_indexes] * cos_theta)

    return batch


def random_scale(batch):
    x_indexes = np.array(range(0, batch.shape[-1], 3))
    y_indexes = np.array(range(1, batch.shape[-1], 3))

    scale = 1 + ((random.random()/100) - 0.005)  # (range 0.99-1.01)
    batch[:, :, x_indexes] = batch[:, :, x_indexes] * scale
    batch[:, :, y_indexes] = batch[:, :, y_indexes] * scale

    return batch


def train_augment(batch):
    """
    Augmentations to do when training
    This is where random flips etc. should go
    The input and output will be a torch Tensor
    The augmentations are applied to a whole batch
    """
    #batch = to_np(batch)
    #batch = random_flip(batch)
    #batch = random_scale(batch)
    #batch = random_rotation(batch)
    #batch = to_tensor(batch)

    return batch
