import enum

import torch
from torchvision.ops.boxes import box_iou

from .. import BACKGROUND_IDX
from .loss import IOU_POSITIVE_MATCH_THRESHOLD
from .prediction import (CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
                         NUM_TOP_DETECTIONS, filter_image_boxes_predictions)


@torch.no_grad()
# , device=None):
def count_boxes_that_seem_to_must_be_matched_to_prior_boxes_with_given_scales(priors_scales, boxes):
    # if not boxes.is_floating_point():
    #     raise TypeError("`boxes` must contain coordinates in relative format (xmin, ymin, xmax, ymax)")
    # priors_scales = torch.tensor(model._priors_scales, device=device)
    if boxes.shape[0] == 0:
        raise ValueError('`boxes` must not be empty')
    priors_scales_t = torch.tensor(priors_scales, device=boxes.device)
    priors_areas = priors_scales_t * priors_scales_t
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    boxes_areas = (xmax - xmin) * (ymax - ymin)
    correct_boxes_areas = boxes_areas[boxes_areas > 0]
    prior_to_gt_area_diff = (
        priors_areas.unsqueeze(-1) - correct_boxes_areas).abs()
    priors_idxs, num_matched_boxes = prior_to_gt_area_diff.argmin(
        0).unique(return_counts=True, sorted=True)
    return [priors_scales[idx] for idx in priors_idxs], num_matched_boxes


class MetricsCalculator:
    _instances = []

    def __init__(self, mIoU=False, accuracy=False, mAP=False,
                 gt_boxes_detected=False, mean_num_pos_def_boxes=False,
                 priors_scales_usage=False, priors_scales=None,
                 iou_positive_match_threshold=IOU_POSITIVE_MATCH_THRESHOLD,
                 confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD, topk=NUM_TOP_DETECTIONS,
                 # alpha=LOCATION_LOSS_COEFF,
                 background_idx=BACKGROUND_IDX):  # , device='cpu'):
        # TODO: debug
        # self.__class__._instances.append(self)

        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

        if priors_scales_usage and priors_scales is None:
            raise ValueError(
                f"`priors_scales` must be given when `priors_scales_usage={priors_scales_usage}`")

        self.total_images = 0
        self.total_gt_boxes = 0

        if self.mIoU:
            self.top1_iou_sum = 0
        if self.accuracy:
            self.top1_tp_tn_preds, self.total_cls_preds = 0, 0
        if self.gt_boxes_detected:
            self.num_gt_boxes_covered_with_not_background_preds = 0
        if self.mean_num_pos_def_boxes:
            self.num_positive_default_boxes = 0
        if self.priors_scales_usage:
            self.priors_scales_to_num_gt_boxes = dict.fromkeys(
                priors_scales, 0)
            self.priors_scales_to_num_gt_boxes_covered_with_not_background_preds = dict.fromkeys(
                priors_scales, 0)

    @torch.no_grad()
    def update(self, gt_boxes, gt_labels, raw_boxes, raw_cls_scores,
               filtered_boxes=None, filtered_boxes_labels=None, filtered_boxes_probs=None,
               num_gt_boxes_covered_with_not_background_preds=None, num_positive_default_boxes=None, gt_boxes_covered_with_not_background_preds=None):
        if self.gt_boxes_detected and num_gt_boxes_covered_with_not_background_preds is None:
            raise ValueError(
                "`num_gt_boxes_covered_with_not_background_preds` is required to calc `gt_boxes_detected`")
        if self.mean_num_pos_def_boxes and num_positive_default_boxes is None:
            raise ValueError(
                "`gt_boxes_detected` is required to calc `mean_num_pos_def_boxes`")
        if self.priors_scales_usage and gt_boxes_covered_with_not_background_preds is None:
            raise ValueError(
                "`gt_boxes_covered_with_not_background_preds` is required to calc `priors_scales_usage`")

        self.total_images += 1
        self.total_gt_boxes += gt_boxes.shape[0]
        if filtered_boxes is None:
            filtered_boxes, filtered_boxes_labels, filtered_boxes_probs = filter_image_boxes_predictions(
                raw_boxes, raw_cls_scores,
                self.confidence_threshold, self.nms_iou_threshold, self.topk, self.background_idx)

        if self.mIoU or self.accuracy:
            if gt_boxes.shape[0] != 1:
                raise ValueError(
                    "image must have exactly one ground truth box to be "
                    "able used in calculation mIoU or accuracy")
            predictions_empty = filtered_boxes.shape[0] == 0
            if not predictions_empty:
                max_confident_idx = filtered_boxes_probs.argmax(dim=-1)
                max_condfident_box = filtered_boxes[max_confident_idx]
                max_condfident_box_label = filtered_boxes_labels[max_confident_idx]
                iou = box_iou(max_condfident_box.unsqueeze(0), gt_boxes).item()
                if self.mIoU:
                    self.top1_iou_sum += iou
                if self.accuracy and iou > self.iou_positive_match_threshold:
                    self.top1_tp_tn_preds += (gt_labels[0]
                                              == max_condfident_box_label).item()
                    self.total_cls_preds += 1
        if self.mAP:
            raise NotImplementedError
        if self.gt_boxes_detected:
            self.num_gt_boxes_covered_with_not_background_preds += num_gt_boxes_covered_with_not_background_preds
        if self.mean_num_pos_def_boxes:
            self.num_positive_default_boxes += num_positive_default_boxes
        if self.priors_scales_usage:
            if gt_boxes_covered_with_not_background_preds.shape[0] != 0:
                results = count_boxes_that_seem_to_must_be_matched_to_prior_boxes_with_given_scales(
                    self.priors_scales, gt_boxes_covered_with_not_background_preds)
                for prior_scale, num_boxes in zip(*results):
                    self.priors_scales_to_num_gt_boxes_covered_with_not_background_preds[prior_scale] += num_boxes.item(
                    )
            results = count_boxes_that_seem_to_must_be_matched_to_prior_boxes_with_given_scales(
                self.priors_scales, gt_boxes)
            for prior_scale, num_boxes in zip(*results):
                self.priors_scales_to_num_gt_boxes[prior_scale] += num_boxes.item()

    def results(self):
        metrics = {}
        if self.mIoU:
            metrics.update(mIoU=self.top1_iou_sum / self.total_images)
        if self.accuracy:
            metrics.update(accuracy=0 if self.total_cls_preds ==
                           0 else self.top1_tp_tn_preds / self.total_cls_preds)
        if self.mAP:
            raise NotImplementedError
        if self.gt_boxes_detected:
            metrics.update(
                gt_boxes_detected=self.num_gt_boxes_covered_with_not_background_preds / self.total_gt_boxes)
        if self.mean_num_pos_def_boxes:
            metrics.update(
                mean_num_pos_def_boxes=self.num_positive_default_boxes / self.total_images)
        if self.priors_scales_usage:
            priors_scales_usage_report = {}
            for prior_scale in self.priors_scales:
                num_pos_gt_boxes = self.priors_scales_to_num_gt_boxes_covered_with_not_background_preds[
                    prior_scale]
                num_gt_boxes = self.priors_scales_to_num_gt_boxes[prior_scale]
                if isinstance(prior_scale, torch.Tensor):
                    prior_scale = prior_scale.item()
                priors_scales_usage_report[prior_scale] = dict(
                    gt_boxes_detected=num_pos_gt_boxes / num_gt_boxes if num_gt_boxes else 0,
                    gt_boxes_total=num_gt_boxes)
            metrics.update(priors_scales_usage=priors_scales_usage_report)
        return metrics


class MetricsCalculation(enum.Enum):
    neither = None
    train = 'train'
    valid = 'valid'
    both = ('train', 'valid')
