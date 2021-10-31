
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output
from torchvision.transforms import functional as TF
from torchvision.utils import draw_bounding_boxes

from . import LABEL_TO_COLOR, LABEL_TO_TAG, get_path_for_saving_plots

# get_ipython().run_line_magic('matplotlib', 'inline')


def show_image(image: torch.Tensor):
    plt.imshow(image.permute(1, 2, 0).cpu())


def draw_predictions(
        image: torch.Tensor, boxes: torch.Tensor, labels: Optional[torch.Tensor] = None, probs: Optional[torch.Tensor] = None,
        *, label2tag=LABEL_TO_TAG, label2color=LABEL_TO_COLOR, fill_box=False, box_width=1, font=None, font_size_ratio_to_img_height=1 / 17, show_plot=False):
    if labels is not None:
        labels = labels.tolist()
        colors = [label2color.get(l, 'blue') for l in labels]
    else:
        colors = ['green'] * len(boxes)
    if labels is not None:
        labels = [label2tag.get(l, "unknown") for l in labels]
        if probs is not None:
            labels = [f"{l}: {round(float(p), 2)}" for l,
                      p in zip(labels, probs)]

    if not isinstance(image, torch.Tensor):
        raise TypeError("`image` must be `torch.Tensor` instance")
    if image.is_floating_point():
        image = TF.convert_image_dtype(image, torch.uint8)
    if boxes.is_floating_point():
        img_w_h = image.shape[-2:][::-1]
        img_w_h_w_h_t = torch.tensor(img_w_h).repeat(2)
        # convert boxes coordinates to absolute form
        boxes = (boxes * img_w_h_w_h_t).round().type(torch.long)
    img_h = image.shape[-2]
    if font_size_ratio_to_img_height is not None:
        font_size = round(img_h * font_size_ratio_to_img_height)
    else:
        font_size = 10
    image = draw_bounding_boxes(image, boxes, labels=labels, colors=colors,
                                fill=fill_box, width=box_width,
                                font=font, font_size=font_size)
    if show_plot:
        show_image(image)
    return image


def get_plots_template(
        loss=None, mIoU=None, accuracy=None,
        gt_boxes_detected=None, mean_num_pos_def_boxes=None):
    plots = []
    if loss is not None:
        if isinstance(loss, str):
            loss = (loss,)
        plot = {"title": "loss",
                "ylabel": "loss",
                "xlabel": "epoch",
                "y_logscale": True,
                "lines": []}
        plots.append(plot)
        for prefix in loss:
            line = {"metric_name": f"{prefix} loss",
                    "label": f"{prefix} loss",
                    "show_best": True,
                    "best_value_label_fmt": "best epoch ({best_value_idx}) by {label}: {best_value:.3e}"}
            plot['lines'].append(line)
    if any([x is not None for x in (accuracy, mIoU)]):
        plot = {"title": "metrics",
                "ylabel": "metric",
                "xlabel": "epoch",
                "y_logscale": False,
                "lines": []}
        plots.append(plot)
        if mIoU is not None:
            if isinstance(mIoU, str):
                mIoU = (mIoU,)
            for prefix in mIoU:
                line = {"metric_name": f"{prefix} mIoU",
                        "label": f"{prefix} mIoU",
                        "show_best": True,
                        "best_value_label_fmt": "best epoch ({best_value_idx}) by {label}: {best_value:.2f}"}
                plot['lines'].append(line)
        if accuracy is not None:
            if isinstance(accuracy, str):
                accuracy = (accuracy,)
            for prefix in accuracy:
                line = {"metric_name": f"{prefix} accuracy",
                        "label": f"{prefix} accuracy",
                        "show_best": True,
                        "best_value_label_fmt": "best epoch ({best_value_idx}) by {label}: {best_value:.2f}"}
                plot['lines'].append(line)
    if any([x is not None for x in (gt_boxes_detected,)]):
        plot = {"title": "metrics",
                "ylabel": "metric",
                "xlabel": "epoch",
                "y_logscale": False,
                "lines": []}
        plots.append(plot)
        if gt_boxes_detected is not None:
            if isinstance(gt_boxes_detected, str):
                gt_boxes_detected = (gt_boxes_detected,)
            for prefix in gt_boxes_detected:
                line = {"metric_name": f"{prefix} gt_boxes_detected",
                        "label": f"{prefix} gt_boxes_detected",
                        "show_best": True,
                        "best_value_label_fmt": "best epoch ({best_value_idx}) by {label}: {best_value:.2f}"}
                plot['lines'].append(line)
    if any([x is not None for x in (mean_num_pos_def_boxes,)]):
        plot = {"title": "metrics",
                "ylabel": "metric",
                "xlabel": "epoch",
                "y_logscale": False,
                "lines": []}
        plots.append(plot)
        if mean_num_pos_def_boxes is not None:
            if isinstance(mean_num_pos_def_boxes, str):
                mean_num_pos_def_boxes = (mean_num_pos_def_boxes,)
            for prefix in mean_num_pos_def_boxes:
                line = {"metric_name": f"{prefix} mean_num_pos_def_boxes",
                        "label": f"{prefix} mean_num_pos_def_boxes",
                        "show_best": True,
                        "best_value_label_fmt": "best epoch ({best_value_idx}) by {label}: {best_value:.2f}"}
                plot['lines'].append(line)
    return plots


# output_widget = ipywidgets.Output()

def show_plots_legacy(
        train_loss, train_metrics, valid_loss, valid_metrics,
        image=None, predicted_boxes=None, predicted_boxes_labels=None, predicted_boxes_probs=None,
        ground_truth_boxes=None, ground_truth_boxes_labels=None,
        image_plot_title="boxes prediction example of valid set",
        model_name=None, epoch_num: Optional[None] = None, save_to_file=False, clear_prev_output=False, logscale_loss=False, logscale_metrics=False):

    if save_to_file and (model_name is None or epoch_num is None):
        raise ValueError(
            "both `model_name` and `epoch_num` must be not None "
            f"when `save_to_file={save_to_file}`")

    nrows, ncols = 1, 1
    if train_metrics is not None or valid_metrics is not None:
        ncols += 1
    if image is not None:
        if image.ndim != 3:
            raise TypeError("`image` must be 3-dimensional tensor")
        ncols += 1
        if predicted_boxes is not None:
            image = draw_predictions(image, predicted_boxes, predicted_boxes_labels,
                                     predicted_boxes_probs, fill_box=True, box_width=1, show_plot=False)
        if ground_truth_boxes is not None:
            image = draw_predictions(
                image, ground_truth_boxes, ground_truth_boxes_labels, fill_box=True, box_width=2, show_plot=False)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        6 * ncols, 6 * nrows), squeeze=True)

    ax_idx = 0
    ax = axes[ax_idx]
    if logscale_loss:
        ax.set_yscale('log')
    ax.plot(np.arange(1, len(train_loss) + 1),
            train_loss, label="train loss", marker='.')
    best_epoch = np.argmin(train_loss)
    best_value = train_loss[best_epoch]
    best_epoch += 1
    ax.axvline(best_epoch, color='r',
               label=f"best epoch ({best_epoch}) by train loss: {best_value:.3e}")
    ax.legend(loc='best')

    if valid_loss is not None:
        ax.plot(np.arange(1, len(valid_loss) + 1),
                valid_loss, label="valid loss", marker='.')
        best_epoch = np.argmin(valid_loss)
        best_value = valid_loss[best_epoch]
        best_epoch += 1
        ax.axvline(best_epoch, color='r',
                   label=f"best epoch ({best_epoch}) by valid loss: {best_value:.3e}")
        ax.legend(loc='best')
    ax.set_title("loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    if train_metrics is not None or valid_metrics is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        if logscale_metrics:
            ax.set_yscale('log')
        if train_metrics is not None:
            for name, values in train_metrics.items():
                ax.plot(np.arange(1, len(values) + 1), values,
                        label="train {}".format(name), marker='.')
                best_epoch = np.argmax(values)
                best_value = values[best_epoch]
                best_epoch += 1
                ax.axvline(best_epoch, color='r',
                           label=f"best epoch ({best_epoch}) by train {name}: {best_value:.2f}")
        if valid_metrics is not None:
            for name, values in valid_metrics.items():
                ax.plot(np.arange(1, len(values) + 1), values,
                        label="valid {}".format(name), marker='.')
                best_epoch = np.argmax(values)
                best_value = values[best_epoch]
                best_epoch += 1
                ax.axvline(best_epoch, color='r',
                           label=f"best epoch ({best_epoch}) by valid {name}: {best_value:.2f}")
                # ax.axvline(best_epoch, label="best epoch by {}".format(name))
        ax.legend(loc='best')
        ax.set_title("metrics")
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")

    if image is not None:
        ax_idx += 1
        ax = axes[ax_idx]
        ax.imshow(image)
        ax.set_title(image_plot_title)

    suptitle_parts = []
    if model_name is not None:
        suptitle_parts.append(model_name)
    if epoch_num is not None:
        suptitle_parts.append(f"epoch {epoch_num}")
    if suptitle_parts:
        fig.suptitle(": ".join(suptitle_parts))

    # display(output_widget)
    # with output_widget:
    #     if clear_prev_output:
    #         clear_output(wait=False)
    #     display(fig)
    #     show_inline_matplotlib_plots()
    # display(output_widget)

    if save_to_file:
        path = get_path_for_saving_plots(model_name, epoch_num)
        # with open(path, 'wb') as file:
        # fig.savefig(file)
        fig.savefig(path)

    if clear_prev_output:
        clear_output(wait=True)

    plt.show()


def show_plots_new(
        history: dict, plots_template: Optional[list] = None,
        image=None, predicted_boxes=None, predicted_boxes_labels=None, predicted_boxes_probs=None,
        ground_truth_boxes=None, ground_truth_boxes_labels=None,
        image_plot_title=None,
        model_name=None, epoch_num: Optional[int] = None,
        plots_in_row=3, x_values_enum_start=1, save_to_file=False, clear_prev_output=False):

    metrics_history = history['metrics']

    if plots_template is None:
        used_metrics = {}
        for split_metric_name in metrics_history:
            split_name, metric_name = split_metric_name.split(maxsplit=1)
            used_metrics.setdefault(metric_name, []).append(split_name)
        plots_template = get_plots_template(**used_metrics)

    if save_to_file and (model_name is None or epoch_num is None):
        raise ValueError(
            "both `model_name` and `epoch_num` must be not None "
            f"when `save_to_file={save_to_file}`")

    if image is not None:
        if image.ndim != 3:
            raise TypeError("`image` must be 3-dimensional tensor")
        if predicted_boxes is not None:
            image = draw_predictions(
                image, predicted_boxes, predicted_boxes_labels, predicted_boxes_probs,
                fill_box=True, box_width=1, show_plot=False)
        if ground_truth_boxes is not None:
            image = draw_predictions(
                image, ground_truth_boxes, ground_truth_boxes_labels,
                fill_box=True, box_width=2, show_plot=False)
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)

    num_plots = len(plots_template) + (1 if image is not None else 0)
    ncols = min(plots_in_row, num_plots)
    nrows = (num_plots + plots_in_row - 1) // plots_in_row

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        6 * ncols, 6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    for plot, ax in zip(plots_template, axes):
        if plot.get('y_logscale', False):
            ax.set_yscale('log')
        if plot.get('x_logscale', False):
            ax.set_xscale('log')
        title = plot.get('title')
        xlabel = plot.get('xlabel')
        ylabel = plot.get('ylabel')
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        for line in plot['lines']:
            metric_history = metrics_history.get(line['metric_name'])
            if metric_history is None:
                continue
            y_values = metric_history['y_values']
            x_values = metric_history['x_values']
            if x_values is None:
                x_values = np.arange(x_values_enum_start, len(
                    y_values) + x_values_enum_start)
            label = line.get('label')
            ax.plot(x_values, y_values, label=label, marker='.')
            if line.get('show_best', False):
                fn = np.argmin if metric_history['best_value_type'] == 'min' else np.argmax
                best_y_value_idx = fn(y_values)
                best_value = y_values[best_y_value_idx]
                best_value_idx = x_values[best_y_value_idx]
                label_fmt = line.get('best_value_label_fmt')
                if label_fmt is not None:
                    ax.axvline(
                        best_value_idx, color='r',
                        label=label_fmt.format(
                            label=label, best_value_idx=best_value_idx, best_value=best_value))
        ax.legend(loc=plot.get('legend_loc', 'best'))

    if image is not None:
        ax = axes[num_plots - 1]
        ax.imshow(image)
        if image_plot_title is not None:
            ax.set_title(image_plot_title)

    suptitle_parts = []
    if model_name is not None:
        suptitle_parts.append(model_name)
    if epoch_num is not None:
        suptitle_parts.append(f"epoch {epoch_num}")
    if suptitle_parts:
        fig.suptitle(": ".join(suptitle_parts))

    # display(output_widget)
    # with output_widget:
    #     if clear_prev_output:
    #         clear_output(wait=False)
    #     display(fig)
    #     show_inline_matplotlib_plots()
    # display(output_widget)

    if save_to_file:
        path = get_path_for_saving_plots(model_name, epoch_num)
        fig.savefig(path)

    if clear_prev_output:
        clear_output(wait=True)

    plt.show()


def show_plots(*args, **kwargs):
    if not args or 'metrics' not in args[0]:
        show_plots_legacy(*args, **kwargs)
    else:
        show_plots_new(*args, **kwargs)

# def find_metric_in_history(train_history, label):
#  for plot in train_history:
#      for line in plot['lines']:
#          if line['label'] == label:
#              return line
#  return None


def show_train_history_plots(history, model_name=None, epochs=None, **kwargs):
    if 'metrics' not in history:
        train_loss = history['train']['loss']
        valid_loss = history['valid']['loss']
        history_train_metrics = history['train']['metrics']
        history_valid_metrics = history['valid']['metrics']
        train_metrics = history_train_metrics if history_train_metrics else None
        valid_metrics = history_valid_metrics if history_valid_metrics else None
        if epochs is None:
            epochs = len(train_loss)
        show_plots(
            train_loss=train_loss,
            train_metrics=train_metrics,
            valid_loss=valid_loss if valid_loss else None,
            valid_metrics=valid_metrics,
            model_name=model_name,
            epoch_num=epochs,
            **kwargs)
        # logscale_loss=True, logscale_metrics=False)
    else:
        train_loss = history['metrics'].get('train loss')
        if train_loss is not None:
            epochs = len(train_loss['y_values'])
        show_plots(history, model_name=model_name, epoch_num=epochs, **kwargs)
