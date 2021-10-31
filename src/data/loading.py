import os

import torch
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()


def collate_fn(batch):
    collated_images = torch.stack([img for img, _ in batch])
    collated_targets = {}
    for _, target in batch:
        for key, value in target.items():
            try:
                collated_targets[key].append(value)
            except KeyError:
                collated_targets[key] = [value]
    return collated_images, collated_targets


def get_data_loaders(batch_size, num_workers=NUM_WORKERS, train_dataset=None, valid_dataset=None, test_dataset=None):
    train_loader = valid_loader = test_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, shuffle=True)
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, valid_loader, test_loader
