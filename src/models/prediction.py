from typing import Iterable, Union

import torch
from torchvision.ops import batched_nms, box_convert

from .. import BACKGROUND_IDX, BOX_FORMAT

# as in the paper (they seemed to be used in inference time to calculate mAP)
NMS_IOU_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.01
NUM_TOP_DETECTIONS = 200


def filter_image_boxes_predictions(
        boxes: torch.Tensor, cls_scores: torch.Tensor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        topk: int = NUM_TOP_DETECTIONS,
        background_idx: int = BACKGROUND_IDX,
        not_to_filter_out_background_boxes=False):
    """Return filtered boxes, labels, probabilities

    Args:
        boxes: `torch.Tensor` of shape of (N, 4) with N boxes
            and 4 coordinates in format `xyxy`
        cls_scores: `torch.Tensor` of shape of (N, M) with logits
            where M is number of the classes
    """
    cls_probs = cls_scores.softmax(-1)
    box_probs, labels = cls_probs.max(-1)

    if not_to_filter_out_background_boxes:
        not_background_cond = torch.ones_like(labels).bool()
    else:
        not_background_cond = labels != background_idx
    confident_preds_cond = box_probs > confidence_threshold

    not_background_confident_preds_cond = not_background_cond & confident_preds_cond
    confident_boxes = boxes[not_background_confident_preds_cond]
    confident_boxes_probs = box_probs[not_background_confident_preds_cond]
    confident_boxes_labels = labels[not_background_confident_preds_cond]
    proper_boxes_idxs = batched_nms(
        confident_boxes, confident_boxes_probs, confident_boxes_labels, nms_iou_threshold)
    proper_boxes_idxs = proper_boxes_idxs[:topk]
    proper_boxes = confident_boxes[proper_boxes_idxs]
    proper_boxes_probs = confident_boxes_probs[proper_boxes_idxs]
    proper_boxes_labels = confident_boxes_labels[proper_boxes_idxs]

    return proper_boxes, proper_boxes_labels, proper_boxes_probs


@torch.no_grad()
def make_prediction(
        model: torch.nn.Module, data: Union[torch.Tensor, torch.utils.data.DataLoader],
        *, confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD, topk=NUM_TOP_DETECTIONS,
        out_fmt: BOX_FORMAT = 'xyxy', background_idx=BACKGROUND_IDX,
        not_to_filter_out_background_boxes=False, device='cpu'):
    def batch_predict(batch):
        results = []
        batch = batch.to(device)
        batch_boxes, batch_cls_scores = model(batch, out_fmt=out_fmt)
        if out_fmt != 'xyxy':
            batch_boxes = box_convert(
                batch_boxes, in_fmt=out_fmt, out_fmt='xyxy')
        for boxes, cls_scores in zip(batch_boxes, batch_cls_scores):
            # print(f"boxes.shape = {boxes.shape}")# TODO: del
            filtered_boxes, labels, probs = filter_image_boxes_predictions(
                boxes, cls_scores,
                confidence_threshold, nms_iou_threshold, topk, background_idx,
                not_to_filter_out_background_boxes)
            if out_fmt != 'xyxy':
                filtered_boxes = box_convert(
                    filtered_boxes, in_fmt='xyxy', out_fmt=out_fmt)
            results.append((filtered_boxes, labels, probs))
        return results
    was_training = model.training
    model.eval()
    single_image_passed = False
    predictions = []
    unappropriate_data_type_msg = (
        "`data` must be 4-dimensional (image batch) or 3-dimensional (single image) tensor,"
        " or `torch.utils.data.DataLoader` of such tensors")
    if isinstance(data, torch.Tensor):
        if data.ndim == 3:
            single_image_passed = True
            data = data.unsqueeze(0)
        elif data.ndim != 4:
            raise TypeError(unappropriate_data_type_msg)
        predictions.extend(batch_predict(data))
    elif isinstance(data, torch.utils.data.DataLoader):
        data_loader_with_unappropriate_data_type_msg_msg = (
            "got data of unknown type from passed `data` argument of "
            "type `torch.utils.data.DataLoder`, "
            "`torch.Tensor` or `Iterable` containing of the such one expected")
        for d in data:
            if not isinstance(d, torch.Tensor):
                if not isinstance(d, Iterable):
                    raise TypeError(
                        data_loader_with_unappropriate_data_type_msg_msg)
                for batch in d:
                    if isinstance(batch, torch.Tensor):
                        break
                else:
                    raise TypeError(
                        data_loader_with_unappropriate_data_type_msg_msg)
            else:
                batch = d
            predictions.extend(batch_predict(batch))
    else:
        raise TypeError(unappropriate_data_type_msg)
    model.train(was_training)
    return predictions[0] if single_image_passed else predictions
