from itertools import chain
from typing import Iterable, List, Tuple

import torch
import torchvision
from torch import nn
from torchvision.ops import box_convert, clip_boxes_to_image

from .. import BOX_FORMAT


class AuxiliaryConv(nn.Sequential):
    def __init__(
            self, in_channels, out_channels,
            num_channels_decreasing_factor=2, kernel_size=3, bias=True):
        hidden_channels = in_channels // num_channels_decreasing_factor
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels,
                      kernel_size=kernel_size, bias=bias),
            nn.ReLU(inplace=True))
        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels


class SSMBDLikeDetector(nn.Module):
    def __init__(
            self,
            backbone: Iterable[nn.Module], feature_layers_nums: List[int],
            num_aux_layers: int,
            num_classes: int, *,
            priors_aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3),
            extra_priors_aspect_ratios=(1,),
            pred_convs_kwargs=None,
            **kwargs
            # start_scale=0.1,
            # end_scale=0.9
    ):
        super().__init__()

        if pred_convs_kwargs is None:
            pred_convs_kwargs = dict(kernel_size=1, stride=1, padding=0)

        self._feature_layers_nums = tuple(feature_layers_nums)
        self._num_aux_layers = num_aux_layers
        self._num_classes = num_classes
        self._priors_aspect_ratios = priors_aspect_ratios
        self._extra_priors_aspect_ratios = extra_priors_aspect_ratios

        self._num_priors = len(priors_aspect_ratios) + \
            len(extra_priors_aspect_ratios)
        self._priors_scales = [torch.tensor(0.1)] + list(
            torch.linspace(0.2, 0.9, len(
                feature_layers_nums) + num_aux_layers - 1)
        )
        # self._priors_scales = [0.1] + list(
        #     torch.linspace(0.2, 0.9, len(feature_layers_nums) + num_aux_layers - 1)
        # )
        self._define_layers(backbone, pred_convs_kwargs, **kwargs)
        self._default_boxes_cache = {}
        self._init()

    def _define_layers(self, backbone, pred_convs_kwargs, **kwargs):
        self.backbone_convs = nn.ModuleList(backbone)
        self.auxiliary_convs = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        self._backbone_feature_layers = []
        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)
            self.loc_convs.append(nn.Conv2d(in_channels=layer.out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=layer.out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

        prev_layer = backbone[-1]
        for _ in range(self._num_aux_layers):
            aux_conv = AuxiliaryConv(
                in_channels=prev_layer.out_channels,
                out_channels=prev_layer.out_channels)
            prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            self.loc_convs.append(nn.Conv2d(in_channels=aux_conv.out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=aux_conv.out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

    @torch.no_grad()
    def _init(self):
        for layer in chain(self.auxiliary_convs, self.loc_convs, self.clf_convs):
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(
                        module.weight, gain=nn.init.calculate_gain('relu'))
                    module.bias.zero_()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def priors_scales(self):
        return self._priors_scales

    def _generate_default_boxes(self, feature_maps_shapes: List[Tuple[int, int]]):
        if len(feature_maps_shapes) != len(self._priors_scales):
            raise ValueError(
                f"length of `feature_maps_shapes` must be equal to {len(self._priors_scales)} "
                "(number of layers used to make predictions).")
        shapes = tuple([tuple(shape[-2:]) for shape in feature_maps_shapes])
        device = self.device

        cached = self._default_boxes_cache.get(shapes)
        if cached is not None:
            cached = self._default_boxes_cache[shapes] = [
                feature_map_boxes.to(device) for feature_map_boxes in cached
            ]
            return list(cached)

        boxes = self._default_boxes_cache[shapes] = []
        for (h, w), scale, subsequent_scale in zip(shapes,
                                                   self._priors_scales,
                                                   self._priors_scales[1:] + [1]):
            feature_map_boxes = []

            cell_half_x = 0.5 / w
            centers_x = torch.linspace(
                cell_half_x, 1 - cell_half_x, w, device=device)
            cell_half_y = 0.5 / h
            centers_y = torch.linspace(
                cell_half_y, 1 - cell_half_y, h, device=device)
            # 1st returned value (generated from 1st argument) is chaging over 1st dimension,
            # 2nd returned value (generated from 2nd argument) is chaging over 2nd dimension:
            grid_cy, grid_cx = torch.meshgrid(centers_y, centers_x)

            for ar in self._priors_aspect_ratios:
                ar_root = ar ** 0.5
                grid_w = torch.full((h, w), scale * ar_root, device=device)
                grid_h = torch.full((h, w), scale / ar_root, device=device)
                feature_map_boxes.append(torch.stack(
                    [grid_cx, grid_cy, grid_w, grid_h], dim=-1))

            extra_scale = (scale * subsequent_scale) ** 0.5
            for ar in self._extra_priors_aspect_ratios:
                ar_root = ar ** 0.5
                grid_w = torch.full((h, w), extra_scale *
                                    ar_root, device=device)
                grid_h = torch.full((h, w), extra_scale /
                                    ar_root, device=device)
                feature_map_boxes.append(torch.stack(
                    [grid_cx, grid_cy, grid_w, grid_h], dim=-1))

            # from list of tensors of shape of (features dim #1, features dim #2, 4) ->
            # to tensor of shape of (features dim #1, features dim #2, num priors, 4):
            feature_map_boxes = torch.stack(feature_map_boxes, dim=-2)

            # clip the priors which overshooted the edges:
            feature_map_boxes = box_convert(
                feature_map_boxes, 'cxcywh', 'xyxy')
            feature_map_boxes = clip_boxes_to_image(
                feature_map_boxes, size=(1.0, 1.0))
            feature_map_boxes = box_convert(
                feature_map_boxes, 'xyxy', 'cxcywh')

            boxes.append(feature_map_boxes)
        return list(boxes)

    def _predict_offsets_and_scores_impl(self, input):
        boxes_gcx_gcy_gw_gh = []
        boxes_cls_scores = []
        result_conv_num = 0
        features = input
        for conv in self.backbone_convs:
            features = conv(features)
            if conv in self._backbone_feature_layers:
                boxes_gcx_gcy_gw_gh.append(
                    self.loc_convs[result_conv_num](features))
                boxes_cls_scores.append(
                    self.clf_convs[result_conv_num](features))
                result_conv_num += 1
        for conv, loc_conv, clf_conv in zip(self.auxiliary_convs,
                                            self.loc_convs[result_conv_num:],
                                            self.clf_convs[result_conv_num:]):
            features = conv(features)
            boxes_gcx_gcy_gw_gh.append(loc_conv(features))
            boxes_cls_scores.append(clf_conv(features))
        return boxes_gcx_gcy_gw_gh, boxes_cls_scores

    def forward(self, input, out_fmt: BOX_FORMAT = 'xyxy'):
        *_, h, w = input.shape

        boxes_gcx_gcy_gw_gh, boxes_cls_scores = self._predict_offsets_and_scores_impl(
            input)

        default_boxes_cx_cy_w_h = self._generate_default_boxes(
            [feature_map_boxes.shape[-2:]
                for feature_map_boxes in boxes_gcx_gcy_gw_gh]
        )
        for i, (defaults, offsets, scores) in enumerate(zip(default_boxes_cx_cy_w_h,
                                                            boxes_gcx_gcy_gw_gh,
                                                            boxes_cls_scores)):
            # from shape of (batch size, num priors * 4, features dim #0, features dim #1) ->
            # to shape of (batch size, num boxes, 4)
            # where num priors is the number of priors per cell and
            # features dim #0 * features dim #1 * num priors is the number of
            # boxes per feature map:
            boxes_gcx_gcy_gw_gh[i] = offsets.permute(
                0, 2, 3, 1).reshape(offsets.shape[0], -1, 4)
            # reshape related default boxes tensor to the same shape:
            default_boxes_cx_cy_w_h[i] = defaults.view(-1, 4)
            # from shape of (batch size, num priors * num classes, features dim #0, features #1) ->
            # to shape of (batch size, num boxes, num classes):
            boxes_cls_scores[i] = scores.permute(0, 2, 3, 1).reshape(
                scores.shape[0], -1, self._num_classes)

        # print(f"{default_boxes_cx_cy_w_h[-1].shape = }; {boxes_cls_scores[-1].shape = }")  # TODO: del
        default_boxes_cx_cy_w_h = torch.cat(
            default_boxes_cx_cy_w_h, dim=0)  # no batch dim
        boxes_gcx_gcy_gw_gh = torch.cat(boxes_gcx_gcy_gw_gh, dim=1)
        boxes_cls_scores = torch.cat(boxes_cls_scores, dim=1)

        def_cx, def_cy, def_w, def_h = default_boxes_cx_cy_w_h.unbind(-1)
        gcx, gcy, gw, gh = boxes_gcx_gcy_gw_gh.unbind(-1)

        # offset the default boxes by the predicted values:
        cx = def_cx + def_w * gcx
        cy = def_cy + def_h * gcy
        w = def_w * gw.exp()
        h = def_h * gh.exp()

        # TODO: debug, turn off offsets to check overlaping default boxes with gt boxes
        # cx = def_cx.expand(input.shape[0], *def_cx.shape) + 0
        # cy = def_cy.expand(input.shape[0], *def_cy.shape) + 0
        # w = def_w.expand(input.shape[0], *def_w.shape) * 1
        # h = def_h.expand(input.shape[0], *def_h.shape) * 1

        boxes_cx_cy_w_h = torch.stack((cx, cy, w, h), dim=-1)
        # print(f"boxes_cx_cy_w_h.shape = {boxes_cx_cy_w_h.shape}")
        # print(f"boxes_cx_cy_w_h.shape = {boxes_cx_cy_w_h.shape}; boxes_cls_scores.shape = {boxes_cls_scores.shape}") # TODO: del
        # return boxes_cx_cy_w_h, boxes_cls_scores, default_boxes_cx_cy_w_h  # TODO: del 3rd
        boxes = boxes_cx_cy_w_h if out_fmt == 'cxcywh' else box_convert(
            boxes_cx_cy_w_h, 'cxcywh', out_fmt)
        # if out_fmt == 'cxcywh':
        #     boxes = boxes_cx_cy_w_h
        # else:
        #     boxes = box_convert(boxes_cx_cy_w_h, 'cxcywh', out_fmt)
        return boxes, boxes_cls_scores

    def __str__(self):
        return (f"{self.__class__.__name__}<"
                f"feature_layers_nums={self._feature_layers_nums}, "
                f"num_aux_layers={self._num_aux_layers}, "
                f"num_classes={self._num_classes}>")


def build_mobilenetv3_small_based_ssd_detector(
        network: torchvision.models.mobilenetv3.MobileNetV3, num_classes):
    return SSMBDLikeDetector(network.features, feature_layers_nums=(3, 8, 12),
                             num_aux_layers=3, num_classes=num_classes)
