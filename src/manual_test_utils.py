import random
from typing import Callable

import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_convert, box_iou
from tqdm.auto import tqdm

from .models.loss import IOU_POSITIVE_MATCH_THRESHOLD
from .models.prediction import make_prediction
from .visualization import draw_predictions, show_image


@torch.no_grad()
def show_transformed_image_with_gt_compared_to_original(tested_transforms: Callable, dataset_with_default_transform, idx=None):
    dataset = dataset_with_default_transform
    original_transforms = dataset._transform

    if idx is None:
        idx = random.randrange(len(dataset))

    print(
        f"Transformed / original images with boxes for {idx}-th image of the dataset.")

    original_tf_image, original_tf_target = dataset[idx]
    original_tf_gt_boxes = original_tf_target['boxes']

    dataset._transform = tested_transforms
    tested_tf_image, tested_tf_target = dataset[idx]
    tested_tf_gt_boxes = tested_tf_target['boxes']
    dataset._transform = original_transforms

    tested_tf_image = draw_predictions(
        tested_tf_image, tested_tf_gt_boxes, fill_box=True, box_width=2, show_plot=False)
    original_tf_image = draw_predictions(
        original_tf_image, original_tf_gt_boxes, fill_box=True, box_width=2, show_plot=False)

    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), squeeze=False)
    axes = axes.flatten()
    axes[0].imshow(tested_tf_image.permute(1, 2, 0))
    axes[0].set_title("tested transforms")
    axes[1].imshow(original_tf_image.permute(1, 2, 0))
    axes[1].set_title("default transforms")
    plt.show()

    return idx, tested_tf_image, tested_tf_target


def _get_device(model, device):
    if device is None:
        try:
            device = model.device
        except AttributeError as exc:
            raise ValueError(
                "either `device` must not be None or "
                "`model` must have `device` attribute") from exc
    return torch.device(device)


@torch.no_grad()
def make_and_show_prediction(model, dataset, idx=None, *,
                             show_ground_truth_labels=False, device=None, **make_prediction_kwargs):
    device = _get_device(model, device)
    if idx is None:
        idx = random.randrange(len(dataset))
    print(f"Trying to predict boxes for {idx}-th image of the dataset.")
    image, target = dataset[idx]
    original_image = dataset.get_original_image(idx).cpu()
    gt_boxes = target['boxes'].cpu()
    if show_ground_truth_labels:
        gt_labels = target['labels'].cpu()
    else:
        gt_labels = None
    image = image.to(device)
    prediction = boxes, labels, probs = make_prediction(
        model, image, out_fmt='xyxy', device=device, **make_prediction_kwargs)
    boxes = boxes.cpu()
    labels = labels.cpu()
    probs = probs.cpu()

    image_with_preds = draw_predictions(
        original_image, boxes, labels, probs, fill_box=True, box_width=1, show_plot=False)
    image_with_preds_and_gt = draw_predictions(
        image_with_preds, gt_boxes, gt_labels, fill_box=True, box_width=2, show_plot=False)
    show_image(image_with_preds_and_gt)

    return idx, prediction


@torch.no_grad()
def find_images_with_no_positive_matches_to_default_boxes(
        model, dataset, start_idx=0, *, layer_num=None,
        iou_positive_match_threshold=IOU_POSITIVE_MATCH_THRESHOLD,
        shapes=None, images_have_same_shape=True, device=None):
    device = _get_device(model, device)
    no_positive_match_images = []
    positive_match_images = []
    was_training = model.training
    model.eval()
    for idx in tqdm(range(start_idx, len(dataset))):
        image, target = dataset[idx]
        gt_boxes = target['boxes'].to(device)
        if shapes is not None:
            default_boxes = model._generate_default_boxes(shapes)
        else:
            if not images_have_same_shape or not model._default_boxes_cache:
                model._default_boxes_cache.clear()
                batch = image.unsqueeze(0).to(device)
                model(batch)
            default_boxes = list(model._default_boxes_cache.values())[0]
        if layer_num is not None:
            default_boxes = default_boxes[layer_num].view(-1, 4)
        else:
            default_boxes = torch.vstack(
                [db.view(-1, 4) for db in default_boxes])
        default_boxes = box_convert(default_boxes, 'cxcywh', 'xyxy')
        gt_vs_df_iou = box_iou(gt_boxes, default_boxes).squeeze(0)
        positive_matches_cond = gt_vs_df_iou > iou_positive_match_threshold
        if not positive_matches_cond.any():
            no_positive_match_images.append(idx)
        else:
            positive_match_images.append(idx)
    model.train(was_training)
    return no_positive_match_images, positive_match_images
