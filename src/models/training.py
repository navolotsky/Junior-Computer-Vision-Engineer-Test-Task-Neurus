from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from tqdm.auto import tqdm

from .. import BACKGROUND_IDX
from ..checkpointing import (convert_legacy_train_history,
                             prepare_history_storage,
                             save_states_and_train_info)
from ..visualization import get_plots_template, show_plots
from .loss import LOCATION_LOSS_COEFF, compute_image_multibox_loss
from .metrics import MetricsCalculation, MetricsCalculator
from .prediction import (CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD,
                         NUM_TOP_DETECTIONS, filter_image_boxes_predictions)
from .utils import is_there_nan_params_named


def train_epoch(
        model, data_loader,
        loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls, optimizer,
        *, max_grad_norm=None, calc_mIoU=False, calc_accuracy=False, calc_mAP=False,
        calc_gt_boxes_detected=False, calc_mean_num_pos_def_boxes=False,
        report_priors_scales_usage=False,
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD, topk=NUM_TOP_DETECTIONS,
        alpha=LOCATION_LOSS_COEFF, background_idx=BACKGROUND_IDX, device='cpu'):

    was_training = model.training
    model.train()

    epoch_loss_sum = 0
    total_images = 0

    metrics_calc = MetricsCalculator(
        mIoU=calc_mIoU, accuracy=calc_accuracy, mAP=calc_mAP,
        gt_boxes_detected=calc_gt_boxes_detected,
        mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes,
        priors_scales_usage=report_priors_scales_usage,
        priors_scales=None if not report_priors_scales_usage else model.priors_scales,
        confidence_threshold=confidence_threshold, nms_iou_threshold=nms_iou_threshold, topk=topk,
        # alpha=LOCATION_LOSS_COEFF,
        background_idx=background_idx  # , device=device
    )

    for batch in tqdm(data_loader, desc="training", leave=False):
        images, targets = batch
        images = images.to(device)

        optimizer.zero_grad()
        batch_boxes_xmin_ymin_xmax_ymax, batch_cls_scores = model(
            images, out_fmt='xyxy')

        batch_losses = []
        for boxes, gt_boxes, cls_scores, gt_labels in zip(batch_boxes_xmin_ymin_xmax_ymax,
                                                          targets['boxes'],
                                                          batch_cls_scores,
                                                          targets['labels']):
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)
            # print(f"gt_boxes.shape = {gt_boxes.shape} & gt_labels.shape = {gt_labels.shape} before, ", end='') # TODO: del, this is for debug
            # gt_boxes = gt_boxes.repeat(3, 1) # TODO: del, this is for debug
            # gt_labels = gt_labels.repeat(3) # TODO: del, this is for debug
            # print(f"for debug we fake it: gt_boxes.shape = {gt_boxes.shape} & gt_labels.shape = {gt_labels.shape}") # TODO: del, this is for debug
            image_loss, num_gt_boxes_covered_with_not_background_preds, num_positive_default_boxes, gt_boxes_covered_with_not_background_preds = compute_image_multibox_loss(
                boxes, gt_boxes, cls_scores, gt_labels,
                loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls,
                return_num_gt_boxes_covered_with_not_background_preds=calc_gt_boxes_detected,
                return_num_positive_default_boxes=calc_mean_num_pos_def_boxes,
                return_gt_boxes_covered_with_not_background_preds=report_priors_scales_usage,
                alpha=alpha, background_idx=background_idx)
            batch_losses.append(image_loss)
            epoch_loss_sum += image_loss.item()
            total_images += 1

            metrics_calc.update(
                gt_boxes, gt_labels, boxes, cls_scores,
                num_gt_boxes_covered_with_not_background_preds=num_gt_boxes_covered_with_not_background_preds,
                num_positive_default_boxes=num_positive_default_boxes,
                gt_boxes_covered_with_not_background_preds=gt_boxes_covered_with_not_background_preds)

        batch_loss = torch.cat(batch_losses).mean()
        if batch_loss.isnan():
            raise RuntimeError(
                "batch loss got a not-a-number value:\n"
                f"batch_losses = {batch_losses}")

        # [sometimes happens] all images in the batch have no positive matches
        # so go to the next:
        if not batch_loss.requires_grad:
            print(f"all images in the batch have no positive matches")
            continue
        batch_loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        check_res = is_there_nan_params_named(model)
        if check_res is not None:
            raise RuntimeError(f"Name of param with nan: {check_res}")

    metrics = metrics_calc.results()
    epoch_mean_loss = epoch_loss_sum / total_images
    metrics.update(loss=epoch_mean_loss)
    model.train(was_training)
    return metrics


@torch.no_grad()
def evaluate(
        model, data_loader,
        loc_criterion_cls=None, conf_pos_critertion_cls=None, conf_neg_critertion_cls=None,
        *, test=False, calc_mIoU=False, calc_accuracy=False, calc_mAP=False,
        calc_gt_boxes_detected=False, calc_mean_num_pos_def_boxes=False,
        report_priors_scales_usage=False,
        prediction_example_choosing: Literal['first',
                                             'last', 'random', 'none'] = 'none',
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD, topk=NUM_TOP_DETECTIONS,
        alpha=LOCATION_LOSS_COEFF, background_idx=BACKGROUND_IDX, device='cpu'):

    loss_clss_none = [x is None for x in (
        loc_criterion_cls, conf_pos_critertion_cls, conf_pos_critertion_cls)]
    if any(loss_clss_none) and not all(loss_clss_none):
        raise ValueError(
            "either all of "
            "`loc_criterion_cls`, "
            "`conf_pos_critertion_cls`, "
            "`conf_pos_critertion_cls` "
            "args must be None or no one must"
        )
    if (
        calc_gt_boxes_detected or
        calc_mean_num_pos_def_boxes or
        report_priors_scales_usage
    ):
        if any(loss_clss_none):
            raise ValueError(
                "all of "
                "`loc_criterion_cls`, "
                "`conf_pos_critertion_cls`, "
                "`conf_pos_critertion_cls` "
                "args must be given when at least one of the `calc_gt_boxes_detected`, "
                "`calc_mean_num_pos_def_boxes` "
                "`report_priors_scales_usage` params set to True"
            )
        else:
            calc_loss_as_by_product = True
    else:
        calc_loss_as_by_product = False

    calc_loss = not all(loss_clss_none)

    was_training = model.training
    model.eval()

    epoch_loss_sum = 0
    total_images = 0

    metrics_calc = MetricsCalculator(
        mIoU=calc_mIoU, accuracy=calc_accuracy, mAP=calc_mAP,
        gt_boxes_detected=calc_gt_boxes_detected,
        mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes,
        priors_scales_usage=report_priors_scales_usage,
        priors_scales=None if not report_priors_scales_usage else model.priors_scales,
        confidence_threshold=confidence_threshold, nms_iou_threshold=nms_iou_threshold, topk=topk,
        # alpha=LOCATION_LOSS_COEFF,
        background_idx=background_idx  # , device=device
    )

    prediction_example = None
    prediction_example_image_num = torch.randint(
        len(data_loader.dataset), (1,)).item()

    for batch in tqdm(data_loader, desc="testing" if test else "validation", leave=False):
        images, targets = batch
        images = images.to(device)

        batch_boxes_xmin_ymin_xmax_ymax, batch_cls_scores = model(
            images, out_fmt='xyxy')

        for boxes, gt_boxes, cls_scores, gt_labels, image_idx in zip(batch_boxes_xmin_ymin_xmax_ymax,
                                                                     targets['boxes'],
                                                                     batch_cls_scores,
                                                                     targets['labels'],
                                                                     targets['idx']):
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)
            # print(f"gt_boxes.shape = {gt_boxes.shape} & gt_labels.shape = {gt_labels.shape} before, ", end='') # TODO: del, this is for debug
            # gt_boxes = gt_boxes.repeat(3, 1) # TODO: del, this is for debug
            # gt_labels = gt_labels.repeat(3) # TODO: del, this is for debug
            # print(f"for debug we fake it: gt_boxes.shape = {gt_boxes.shape} & gt_labels.shape = {gt_labels.shape}") # TODO: del, this is for debug
            if calc_loss or calc_loss_as_by_product:
                image_loss, num_gt_boxes_covered_with_not_background_preds, num_positive_default_boxes, gt_boxes_covered_with_not_background_preds = compute_image_multibox_loss(
                    boxes, gt_boxes, cls_scores, gt_labels,
                    loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls,
                    return_num_gt_boxes_covered_with_not_background_preds=calc_gt_boxes_detected,
                    return_num_positive_default_boxes=calc_mean_num_pos_def_boxes,
                    return_gt_boxes_covered_with_not_background_preds=report_priors_scales_usage,
                    alpha=alpha, background_idx=background_idx)
                epoch_loss_sum += image_loss.item()

            image_boxes_predictions_filtered = filter_image_boxes_predictions(
                boxes, cls_scores,
                confidence_threshold, nms_iou_threshold, topk, background_idx)
            image_boxes, image_boxes_labels, image_boxes_probs = image_boxes_predictions_filtered
            predictions_empty = image_boxes.shape[0] == 0
            if predictions_empty:
                image_boxes = image_boxes_labels = image_boxes_probs = None

            metrics_calc.update(
                gt_boxes, gt_labels, boxes, cls_scores,
                *image_boxes_predictions_filtered,
                num_gt_boxes_covered_with_not_background_preds=num_gt_boxes_covered_with_not_background_preds,
                num_positive_default_boxes=num_positive_default_boxes,
                gt_boxes_covered_with_not_background_preds=gt_boxes_covered_with_not_background_preds)

            if (
                prediction_example is None and
                (
                    prediction_example_choosing == 'first' or
                    (
                        prediction_example_choosing == 'random' and
                        total_images == prediction_example_image_num
                    )
                )
            ):
                image = data_loader.dataset.get_original_image(
                    image_idx).to(device)
                prediction_example = [
                    image, image_boxes, image_boxes_labels, image_boxes_probs, gt_boxes, gt_labels]

            total_images += 1

    if (
        prediction_example_choosing == 'last' or
        (prediction_example_choosing == 'random' and prediction_example is None)
    ):
        image = data_loader.dataset.get_original_image(image_idx).to(device)
        prediction_example = [
            image, image_boxes, image_boxes_labels, image_boxes_probs, gt_boxes, gt_labels]

    metrics = metrics_calc.results()
    if calc_loss:
        epoch_mean_loss = epoch_loss_sum / total_images
        metrics.update(loss=epoch_mean_loss)
    model.train(was_training)
    return metrics, prediction_example


def get_priors_scales_usage_report_text(priors_scales_usage, split, epoch_num):
    priors_scales_usage_report = ""
    priors_scales_usage_report += f"priors scales usage report for {split} (epoch {epoch_num}):\n"
    priors_scales_usage_report += '-' * 11 + \
        '---' + '-' * 17 + '---' + '-' * 14 + '\n'
    priors_scales_usage_report += "prior scale | gt boxes detected | gt boxes total\n"
    priors_scales_usage_report += '-' * 11 + \
        '---' + '-' * 17 + '---' + '-' * 14 + '\n'
    priors_scales_usage_report += "\n".join(
        f"{scale:11.2f} | {usage['gt_boxes_detected']:17.2f} | {int(usage['gt_boxes_total']):14,d}"
        for scale, usage in priors_scales_usage.items()
    ) + '\n'
    priors_scales_usage_report += '-' * 11 + \
        '---' + '-' * 17 + '---' + '-' * 14 + '\n'
    return priors_scales_usage_report


def train(
        history: List[Dict],
        model_name, model, num_epochs, train_loader,
        loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls,
        optimizer, valid_loader, *,
        lr_scheduler=None, lr_scheduler_1st_epoch=None, use_valid_loss_for_scheduler=False, init_epoch_callback=None, max_grad_norm=None,
        calc_mIoU=MetricsCalculation.neither, calc_accuracy=MetricsCalculation.neither,
        calc_gt_boxes_detected=MetricsCalculation.neither, calc_mean_num_pos_def_boxes=MetricsCalculation.neither,
        report_priors_scales_usage=MetricsCalculation.neither,
        save_at_every_epoch=True,
        prediction_example_choosing: Literal['first',
                                             'last', 'random', 'none'] = 'first',
        show_only_last_plot=True, save_plots=True,
        early_stopping=False, es_patience=5, es_rel_threshold=5e-2, use_valid_loss_for_es=True,
        confidence_threshold=CONFIDENCE_THRESHOLD, nms_iou_threshold=NMS_IOU_THRESHOLD, topk=NUM_TOP_DETECTIONS,
        alpha=LOCATION_LOSS_COEFF, background_idx=BACKGROUND_IDX, device='cpu', resume=False):
    TRAIN_PREFIX = 'train'
    VALID_PREFIX = 'valid'

    calc_mIoU = MetricsCalculation(calc_mIoU)
    calc_accuracy = MetricsCalculation(calc_accuracy)
    calc_gt_boxes_detected = MetricsCalculation(calc_gt_boxes_detected)
    calc_mean_num_pos_def_boxes = MetricsCalculation(
        calc_mean_num_pos_def_boxes)
    report_priors_scales_usage = MetricsCalculation(report_priors_scales_usage)

    train_metric_calculation = (
        MetricsCalculation.train, MetricsCalculation.both)
    valid_metric_calculation = (
        MetricsCalculation.valid, MetricsCalculation.both)

    if not isinstance(history, dict):
        raise TypeError("`history` must be `dict` instance")
    if history and not resume:
        raise ValueError(
            f"`history` must be empty dict when `resume={resume}` is given")
    elif not history and resume:
        raise ValueError(
            f"`history` must be non-empty dict when `resume={resume}` is given")

    if 'metrics' not in history:
        converted_history = convert_legacy_train_history(history)
        history.clear()
        history.update(converted_history)

    prepare_history_storage(
        history,
        loss=(TRAIN_PREFIX, VALID_PREFIX),
        mIoU=calc_mIoU.value,
        accuracy=calc_accuracy.value,
        gt_boxes_detected=calc_gt_boxes_detected.value,
        mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes.value,
        priors_scales_usage=report_priors_scales_usage.value)

    plots_template = get_plots_template(
        loss=(TRAIN_PREFIX, VALID_PREFIX),
        mIoU=calc_mIoU.value,
        accuracy=calc_accuracy.value,
        gt_boxes_detected=calc_gt_boxes_detected.value,
        mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes.value)

    if lr_scheduler_1st_epoch is None:
        lr_scheduler_1st_epoch = 1

    train_loss_history = history['metrics'][f'{TRAIN_PREFIX} loss']['y_values']
    valid_loss_history = history['metrics'][f'{VALID_PREFIX} loss']['y_values']

    min_valid_loss = min([float('inf')] + valid_loss_history)

    es_prev_valid_loss = None
    es_num_epochs_without_improvements = 0

    start_epoch = len(train_loss_history) + 1
    for epoch_num in tqdm(range(start_epoch, start_epoch + num_epochs), desc="epochs"):
        if init_epoch_callback is not None:
            init_epoch_callback(epoch_num)

        train_metrics = train_epoch(
            model, train_loader, loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls, optimizer,
            max_grad_norm=max_grad_norm,
            calc_mIoU=calc_mIoU in train_metric_calculation,
            calc_accuracy=calc_mIoU in train_metric_calculation,
            calc_gt_boxes_detected=calc_gt_boxes_detected in train_metric_calculation,
            calc_mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes in train_metric_calculation,
            report_priors_scales_usage=report_priors_scales_usage in train_metric_calculation,
            alpha=alpha, background_idx=background_idx, device=device)
        train_loss = train_metrics['loss']

        valid_metrics, prediction_example = evaluate(
            model, valid_loader, loc_criterion_cls, conf_pos_critertion_cls, conf_neg_critertion_cls,
            test=False,
            calc_mIoU=calc_mIoU in valid_metric_calculation,
            calc_accuracy=calc_mIoU in valid_metric_calculation,
            calc_gt_boxes_detected=calc_gt_boxes_detected in valid_metric_calculation,
            calc_mean_num_pos_def_boxes=calc_mean_num_pos_def_boxes in valid_metric_calculation,
            report_priors_scales_usage=report_priors_scales_usage in valid_metric_calculation,
            prediction_example_choosing=prediction_example_choosing,
            confidence_threshold=confidence_threshold, nms_iou_threshold=nms_iou_threshold, topk=topk,
            alpha=alpha, background_idx=background_idx, device=device)
        valid_loss = valid_metrics['loss']

        if lr_scheduler is not None and epoch_num >= lr_scheduler_1st_epoch:
            lr_scheduler.step(
                valid_loss if use_valid_loss_for_scheduler else train_loss)

        priors_scales_usage_reports = ""
        for split, metrics in zip([TRAIN_PREFIX, VALID_PREFIX], [train_metrics, valid_metrics]):
            priors_scales_usage = metrics.pop('priors_scales_usage', None)
            if priors_scales_usage is not None:
                history['priors_scales_usage'][split][epoch_num] = priors_scales_usage
                # prepare report for print:
                priors_scales_usage_reports += get_priors_scales_usage_report_text(
                    priors_scales_usage, split, epoch_num)
                priors_scales_usage_reports += '\n'

            for label, value in metrics.items():
                label = split + ' ' + label
                metric_history = history['metrics'][label]
                metric_history['y_values'].append(value)
                metric_history['x_values'].append(epoch_num)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            save_states_and_train_info(
                model_name, epoch_num,
                history, model, optimizer, lr_scheduler,
                epoch_type='best')

        if save_at_every_epoch:
            save_states_and_train_info(
                model_name, epoch_num,
                history, model, optimizer, lr_scheduler,
                epoch_type='intermediate')

        if prediction_example is not None:
            image, boxes, labels, probs, gt_boxes, gt_labels = prediction_example
            image = image.cpu()
            gt_boxes = gt_boxes.cpu()
            gt_labels = gt_labels.cpu()
            if boxes is not None:
                boxes = boxes.cpu()
                labels = labels.cpu()
                probs = probs.cpu()
        else:
            image = boxes = labels = probs = gt_boxes = gt_labels = None

        show_plots(
            history, plots_template,
            image=image,
            predicted_boxes=boxes,
            predicted_boxes_labels=labels,
            predicted_boxes_probs=probs,
            ground_truth_boxes=gt_boxes,
            # ground_truth_boxes_labels=gt_labels,
            # ground_truth_boxes_labels=None,
            image_plot_title="boxes prediction example (valid set)",
            model_name=model_name,
            epoch_num=epoch_num,
            save_to_file=save_plots,
            clear_prev_output=show_only_last_plot
        )
        if priors_scales_usage_reports:
            print(priors_scales_usage_reports)

        if early_stopping:
            loss = valid_loss if use_valid_loss_for_es else train_loss
            if es_prev_valid_loss is None:
                es_prev_valid_loss = loss
                es_num_epochs_without_improvements = 0
            elif loss < es_prev_valid_loss * (1 - es_rel_threshold):
                es_prev_valid_loss = loss
                es_num_epochs_without_improvements = 0
            else:
                es_num_epochs_without_improvements += 1
            if es_num_epochs_without_improvements > es_patience:
                print(
                    f"Early stopping at {epoch_num} epoch after {es_num_epochs_without_improvements} "
                    f"epochs without improvements.\nCurrent epoch loss: {loss}, "
                    f"Previous loss after significant change: {es_prev_valid_loss}.\n"
                    f"Used loss: {VALID_PREFIX if use_valid_loss_for_es else TRAIN_PREFIX}.")
                break

    save_states_and_train_info(
        model_name, epoch_num,
        history, model, optimizer, lr_scheduler,
        epoch_type='last')
