
import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Union

import torch

from . import get_path_for_saving_model


def prepare_history_storage(
        history: list,
        loss=('train', 'valid'), mIoU=('valid',), accuracy=('valid',),
        gt_boxes_detected=None, mean_num_pos_def_boxes=None, priors_scales_usage=None):
    min_metrics_names = ("loss",)
    metrics = history.setdefault('metrics', {})
    for name, splits in zip(
            [
                "loss", "mIoU", "accuracy",
                "gt_boxes_detected", "mean_num_pos_def_boxes"],
            [
                loss, mIoU, accuracy,
                gt_boxes_detected, mean_num_pos_def_boxes]
    ):
        if splits is not None:
            if isinstance(splits, str):
                splits = (splits,)
            for prefix in splits:
                full_name = f"{prefix} {name}"
                if full_name in metrics:
                    continue
                metrics[full_name] = {
                    "y_values": [], "x_values": [],
                    "best_value_type": "min" if name in min_metrics_names else "max"
                }
    if priors_scales_usage is not None:
        if isinstance(priors_scales_usage, str):
            priors_scales_usage = (priors_scales_usage,)
        priors_scales_usage_history = history.setdefault(
            'priors_scales_usage', {})
        for prefix in priors_scales_usage:
            priors_scales_usage_history.setdefault(prefix, {})


def convert_legacy_train_history(legacy_train_history, x_values_enum_start=1):
    metrics = {}
    for split, content in legacy_train_history.items():
        if 'loss' in content:
            prefixes = metrics.setdefault('loss', [])
            prefixes.append(split)
        for metric in content['metrics']:
            prefixes = metrics.setdefault(metric, [])
            prefixes.append(split)
    converted_history = {}
    prepare_history_storage(converted_history, **metrics)
    for split, content in legacy_train_history.items():
        if 'loss' in content:
            copied_values = content['loss']
            label = split + ' ' + 'loss'
            converted_metric_history = converted_history['metrics'][label]
            converted_metric_history['best_value_type'] = 'min'
            converted_metric_history['y_values'].extend(copied_values)
            converted_metric_history['x_values'].extend(
                list(range(x_values_enum_start,
                     x_values_enum_start + len(copied_values)))
            )
        for metric, copied_values in content['metrics'].items():
            label = split + ' ' + metric
            converted_metric_history = converted_history['metrics'][label]
            converted_metric_history['best_value_type'] = 'max'
            converted_metric_history['y_values'].extend(copied_values)
            converted_metric_history['x_values'].extend(
                list(range(x_values_enum_start,
                     x_values_enum_start + len(copied_values)))
            )
    return converted_history


SAVED_EPOCH_TYPE = Literal['best', 'last', 'intermediate']
LOADED_EPOCH_TYPE = Union[int, Literal['best', 'last']]


def save_states_and_train_info(
        model_name, epoch_num: int, train_history, model, optimizer,
        lr_scheduler=None, epoch_type: SAVED_EPOCH_TYPE = 'intermediate', **kwargs):
    path = get_path_for_saving_model(
        model_name, epoch_num if epoch_type == 'intermediate' else epoch_type)
    states = {
        'model_name': model_name,
        'epoch_num': epoch_num,
        'train_history': train_history,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_scheduler_state': None if lr_scheduler is None else lr_scheduler.state_dict()
    }
    states.update(kwargs)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    torch.save(states, path)


def load_states_and_train_info(model_name, epoch: LOADED_EPOCH_TYPE = 'best', device='cpu'):
    path = get_path_for_saving_model(model_name, epoch)
    return torch.load(path, map_location=device)


def load_state_dicts_and_get_train_history(
        states: dict, model, optimizer=None, lr_scheduler=None, **kwargs):
    train_history = states['train_history']
    model.load_state_dict(states['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(states['optimizer_state'])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(states['lr_scheduler_state'])
    for key, instance in kwargs.items():
        if not hasattr(instance, 'load_state_dict'):
            raise TypeError(
                f"{instance} for `{key}` has no `load_state_dict` method")
        if key not in states:
            raise ValueError(f"`{key}` is not found in state dicts")
    return train_history
