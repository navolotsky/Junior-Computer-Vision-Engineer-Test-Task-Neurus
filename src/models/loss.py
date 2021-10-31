import torch
from torchvision.ops import box_iou

from .. import BACKGROUND_IDX

# as in the paper: https://arxiv.org/abs/1512.02325
IOU_POSITIVE_MATCH_THRESHOLD = 0.5
HARD_NEGATIVE_MINING_NEG_TO_POS_RATIO = 3
LOCATION_LOSS_COEFF = 1


def compute_image_multibox_loss(
        boxes, gt_boxes, cls_scores, gt_labels,
        loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls,
        *,
        return_num_gt_boxes_covered_with_not_background_preds=False,
        return_num_positive_default_boxes=False,
        return_gt_boxes_covered_with_not_background_preds=False,
        alpha=LOCATION_LOSS_COEFF, background_idx=BACKGROUND_IDX):
    def get_returned_values():
        if num_positives == 0:
            return (
                image_loss,
                torch.tensor(
                    0).item() if return_num_gt_boxes_covered_with_not_background_preds else None,
                num_positives if return_num_positive_default_boxes else None,
                gt_boxes[[]] if return_gt_boxes_covered_with_not_background_preds else None)
        if return_num_gt_boxes_covered_with_not_background_preds or return_gt_boxes_covered_with_not_background_preds:
            idxs_matched_positively = idxs_matched[positive_match_cond]
            not_background_match_cond = positive_cls_scores.argmax(
                -1) != background_idx
            gt_boxes_covered_with_not_background_preds_idxs = idxs_matched_positively[not_background_match_cond].unique(
            )
        return (
            image_loss,
            gt_boxes_covered_with_not_background_preds_idxs.shape[
                0] if return_num_gt_boxes_covered_with_not_background_preds else None,
            num_positives if return_num_positive_default_boxes else None,
            gt_boxes[gt_boxes_covered_with_not_background_preds_idxs] if return_gt_boxes_covered_with_not_background_preds else None)
    # it's old_compute_image_multibox_loss because it works better
    device = boxes.device
    loc_criterion = loc_criterion_cls(reduction='sum')
    conf_pos_critertion = conf_pos_critertion_cls(reduction='sum')
    conf_neg_critertion = conf_neg_critertion_cls(reduction='none')

    iou = box_iou(boxes, gt_boxes)
    iou_matched, idxs_matched = iou.max(dim=-1)
    # print(f"iou_matched.shape = {iou_matched.shape} ")
    # print(f"idxs_matched.shape = {idxs_matched.shape} ")
    matched_gt_boxes = gt_boxes[idxs_matched]
    matched_gt_labels = gt_labels[idxs_matched]

    # iou threshold to consider the match as "positive"
    positive_match_cond = iou_matched > IOU_POSITIVE_MATCH_THRESHOLD
    negative_match_cond = ~positive_match_cond

    # print(f"positive_match_cond.shape = {positive_match_cond.shape}; boxes.shape = {boxes.shape}")
    positive_boxes = boxes[positive_match_cond]
    num_positives = positive_boxes.shape[0]
    # no positive matches so according to the paper, set the loss to zero:
    # TODO: debug
    # if num_positives > 100:
    #     print(f"gt_boxes = {gt_boxes}")
    #     print(f"iou = {iou}")
    #     print(f"iou_matched = {iou_matched}")
    #     print(f"positive_match_cond = {positive_match_cond}")
    #     print(f"positive_match_cond.sum() = {positive_match_cond.sum()}")
    #     print(f"iou_matched[positive_match_cond] = {iou_matched[positive_match_cond]}")
    #     print(f"cls_scores[positive_match_cond] = {cls_scores[positive_match_cond]}")
    #     raise Exception("debug")
    if num_positives == 0:
        # TODO: debug
        # raise RuntimeError("no positive matches")
        # non-zero-dimensional float tensor is required to `.cat(batch_losses).mean()` and
        # then `.backward()` because zero-dimensional tensors cannot be concatenated and
        # mean can be calculated only for floating types
        image_loss = torch.tensor([.0]).to(device)
        # or `image_loss = torch.tensor(0, dtype=torch.float).unsqueeze(0)`
        # return image_loss
        return get_returned_values()

    positive_gt_boxes = matched_gt_boxes[positive_match_cond]
    loc_loss = loc_criterion(positive_boxes, positive_gt_boxes)
    # print(f"{loc_loss.item() = }")  # TODO: del

    positive_cls_scores = cls_scores[positive_match_cond]
    positive_gt_labels = matched_gt_labels[positive_match_cond]
    # print(f"positive_cls_scores.shape = {positive_cls_scores.shape}; positive_gt_labels.shape = {positive_gt_labels.shape}")  # TODO: del this
    conf_loss_pos = conf_pos_critertion(
        positive_cls_scores, positive_gt_labels)

    negative_cls_scores = cls_scores[negative_match_cond]
    negative_gt_labels = torch.full(
        negative_cls_scores.shape[:-1], background_idx, device=device)
    negatives_losses = conf_neg_critertion(
        negative_cls_scores, negative_gt_labels)
    num_negatives = negatives_losses.shape[0]
    num_hard_negatives = int(
        num_positives * HARD_NEGATIVE_MINING_NEG_TO_POS_RATIO)
    if num_negatives > num_hard_negatives:
        hard_negatives_losses, _ = negatives_losses.topk(
            num_hard_negatives, largest=True, sorted=False)
    else:
        hard_negatives_losses = negatives_losses
    # print(f"num_positives = {num_positives}; hard_negatives_losses.shape = {hard_negatives_losses.shape}") # TODO: del this
    # multiplied by the ratio because all the three losses are
    # reduced by the same number of positives in the paper:
    # conf_loss_neg = hard_negatives_losses.sum() * HARD_NEGATIVE_MINING_NEG_TO_POS_RATIO
    conf_loss_neg = hard_negatives_losses.sum()

    # print(f"{conf_loss_pos.item() = }; {conf_loss_neg.item() = }; {loc_loss.item() = }") # TODO: del, #2
    # print(f"conf_loss_pos = {conf_loss_pos}; conf_loss_neg = {conf_loss_neg}; loc_loss = {loc_loss}")
    # all the losses are reduced by the same number of positives in the paper:
    image_loss = torch.unsqueeze(
        (conf_loss_pos + conf_loss_neg + alpha * loc_loss) / num_positives, dim=0)
    # if return_num_positives or return_not_gt_boxes_covered_with_not_background_preds:
    #     result = [image_loss, None, None]
    # else:
    #     result = image_loss
    # if return_num_positives:
    #     result[1] = num_positives
    # if return_not_gt_boxes_covered_with_not_background_preds:
    #     gt_boxes_idxs = torch.arange(0, gt_boxes.shape[-1], device=device)
    #     covered_gt_boxes_idxs = idxs_matched.unique()
    #     not_convered_gt_boxes_idxs = torch.tensor(
    #         [idx for idx in gt_boxes_idxs if idx not in covered_gt_boxes_idxs],
    #         device=device)
    #     not_gt_boxes_covered_with_not_background_preds = gt_boxes.index_select(dim=-1, index=not_convered_gt_boxes_idxs)
    #     result[2] = not_gt_boxes_covered_with_not_background_preds
    # return result
    return get_returned_values()
