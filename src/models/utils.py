from typing import Iterable, Union

import torch
import torch.nn as nn

from .base_detector import SSMBDLikeDetector


def set_requires_grad(model: Union[nn.Module, Iterable[nn.Module]], value: bool):
    modules = []
    if isinstance(model, nn.Module):
        modules = [model]
    elif isinstance(model, Iterable):
        modules = list(model)
    if not modules or not all(isinstance(m, nn.Module) for m in modules):
        raise TypeError(
            "`model` must be instance of `nn.Module` or `Iterable` of `nn.Module`")
    for m in modules:
        for param in m.parameters():
            param.requires_grad_(value)


def freeze_all_weights(model):
    set_requires_grad(model, False)


def unfreeze_all_weights(model):
    set_requires_grad(model, True)


def init_epoch(epoch, model, unfreeze_at_epoch):
    if epoch == unfreeze_at_epoch:
        print(f"unfreezed at {epoch}")
        unfreeze_all_weights(model)


def is_there_nan_params(model):
    return any(param.isnan().any() for param in model.parameters())


def is_there_nan_params_named(model):
    for name, param in model.named_parameters():
        if param.isnan().any():
            return name
    return None


def count_num_params(model, only_trainable=True):
    return sum(param.numel() for param in model.parameters() if not only_trainable or param.requires_grad)


@torch.no_grad()
def compare_nets(*instances: nn.Module, print_results=True, input_image_shape=(224, 224), device='cpu'):
    batch = torch.randn(3, *input_image_shape).unsqueeze(0).to(device)
    comparison = []
    for instance in instances:
        num_params = count_num_params(instance)
        if isinstance(instance, SSMBDLikeDetector):
            boxes, scores = instance(batch)
            num_boxes = boxes.shape[1]
            desc = str(instance)
        else:
            num_boxes = None
            desc = ""
        comparison.append(
            dict(class_=instance.__class__.__name__, desc=desc,
                 num_boxes=num_boxes, num_params=num_params)
        )

    if print_results:
        for net in comparison:
            for key, value in net.items():
                if not value:
                    continue
                if isinstance(value, int):
                    value = f"{value:,}"
                print(f"{key}: {value}")
            print("\n---\n")
    return comparison
