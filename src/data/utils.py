import glob
import os

from torch.utils.data import random_split


def gather_data(folder_path, img_ext=".jpg", annot_ext=".txt", sort=True):
    result = []
    for img_path in glob.glob(os.path.join(folder_path, "*" + img_ext)):
        annot_path = img_path[:-len(img_ext)] + annot_ext
        if not os.path.exists(annot_path):
            raise ValueError(f"cannot find annotation file for {img_path}")
        result.append((img_path, annot_path))
    return sorted(result) if sort else result


def random_split_data(data, split_sizes=(0.7,)):
    if any(x < 0 for x in split_sizes):
        raise ValueError("split sizes must be non-negative")
    if not 0 < sum(split_sizes) < 1:
        raise ValueError("split sizes sum must be > 0 and < 1")
    data_len = len(data)
    lengths = [round(data_len * size) for size in split_sizes]
    tail_len = data_len - sum(lengths)
    lengths.append(tail_len)
    splits = random_split(data, lengths=lengths)
    return splits
