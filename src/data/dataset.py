from copy import deepcopy

import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode


class CatsDogsDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=None, load_bbs_in_rel_coords=False, use_cache=True, cache_transformed=False):
        super().__init__()
        self._paths = tuple(paths)
        self._transform = transform
        self._load_bbs_in_rel_coords = load_bbs_in_rel_coords
        self._use_cache = use_cache
        self._cache_transformed = cache_transformed
        if use_cache:
            self._cache = list([None] * len(paths))

    def _load_data(self, idx):
        img_path, annot_path = self._paths[idx]
        # convert if it is grayscale one, otherwise the tensor
        # will have 1 element in channels dimension:
        img = read_image(img_path, mode=ImageReadMode.RGB)
        with open(annot_path, encoding='utf-8') as file:
            annot_lines = [line.strip() for line in file]
        labels, boxes = [], []
        for line in annot_lines:
            if not line:
                continue
            label, xmin, ymin, xmax, ymax = map(int, line.split())
            labels.append(label)
            if self._load_bbs_in_rel_coords:
                *_, img_h, img_w = img.shape
                boxes.append([xmin / img_w, ymin / img_h,
                             xmax / img_w, ymax / img_h])
            else:
                boxes.append([xmin, ymin, xmax, ymax])
        target = dict(labels=labels, boxes=boxes, idx=idx)
        return img, target

    def __getitem__(self, idx):
        result = None

        if self._use_cache:
            result = self._cache[idx]
            if result is not None:
                if self._cache_transformed:
                    return result
                else:
                    result = deepcopy(result)

        if result is None:
            result = self._load_data(idx)
            if self._use_cache and not self._cache_transformed:
                self._cache[idx] = result
                result = deepcopy(result)

        img, target = result
        labels = torch.tensor(target['labels'])
        boxes = torch.tensor(target['boxes'])

        if self._transform is not None:
            img, boxes = self._transform(img, boxes)

        target['labels'] = labels
        target['boxes'] = boxes
        result = (img, target)

        if self._use_cache and self._cache_transformed:
            self._cache[idx] = result
        return result

    def __len__(self):
        return len(self._paths)

    def get_original_image(self, idx):
        return read_image(self._paths[idx][0], mode=ImageReadMode.RGB)
