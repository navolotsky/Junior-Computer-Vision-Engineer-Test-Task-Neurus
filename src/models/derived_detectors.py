

import functools
from itertools import chain
from typing import Iterable, List
import torch

import torchvision.models.mobilenetv3 as mobilenetv3
from torch import nn

from .base_detector import AuxiliaryConv, SSMBDLikeDetector


# === Modified MobileNetV3-Small ===
class OutputConvAppendix(nn.Sequential):
    @property
    def out_channels(self):
        for i in range(len(self) - 1, -1, -1):
            if hasattr(self[i], 'out_channels'):
                return self[i].out_channels
        raise RuntimeError('no `out_channels` found in all the modules')


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector(SSMBDLikeDetector):
    def __init__(
            self,
            backbone: Iterable[nn.Module], feature_layers_nums: List[int],
            # bacbone_feature_layers_appendices: Iterable[nn.Module],
            num_aux_layers: int,
            num_classes: int, *,
            priors_aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3),
            extra_priors_aspect_ratios=(1,),
            # start_scale=0.1,
            # end_scale=0.9
            norm_layer=None,
            aux_convs_decreasing_factor=1,
            aux_convs_bias=False,
            pred_convs_kwargs=dict(kernel_size=1, stride=1, padding=0)
    ):
        backbone = list(backbone)
        if len(backbone) != 13:
            raise ValueError("`backbone` must contain exactly 13 layers")
        if not all([0 <= num < 12 for num in feature_layers_nums]):
            raise ValueError(
                "`feature_layers_nums` must contain ints between 0 (incl) and 12 (excl)")
        super().__init__(
            backbone, feature_layers_nums, num_aux_layers, num_classes,
            priors_aspect_ratios=priors_aspect_ratios, extra_priors_aspect_ratios=extra_priors_aspect_ratios, pred_convs_kwargs=pred_convs_kwargs,
            norm_layer=norm_layer, aux_convs_decreasing_factor=aux_convs_decreasing_factor, aux_convs_bias=aux_convs_bias)

    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if norm_layer is None:
            norm_layer = functools.partial(
                nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError(
                "There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(
                    conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(
                    addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                output_channels = 6 * input_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

        prev_layer = self.backbone_convs[-1]
        for _ in range(self._num_aux_layers):
            aux_conv = AuxiliaryConv(
                in_channels=prev_layer.out_channels,
                out_channels=prev_layer.out_channels,
                num_channels_decreasing_factor=aux_convs_decreasing_factor,
                bias=aux_convs_bias)
            prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            self.loc_convs.append(nn.Conv2d(in_channels=aux_conv.out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=aux_conv.out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

    @property
    def potentially_trained_layers(self):
        return self._potentially_trained_layers

    @torch.no_grad()
    def _init(self):
        for layer in chain(self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        for layer in chain(self.auxiliary_convs, self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(
                        m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        m.bias.zero_()

    def _predict_offsets_and_scores_impl(self, input):
        boxes_gcx_gcy_gw_gh = []
        boxes_cls_scores = []
        result_conv_num = 0
        output = input
        for conv in self.backbone_convs:
            output = conv(output)
            if conv not in self._backbone_feature_layers:
                continue
            if self.backbone_outputs_convs_appendices:
                output_appendix = self.backbone_outputs_convs_appendices[result_conv_num]
                features = output_appendix(output)
            else:
                features = output

            boxes_gcx_gcy_gw_gh.append(
                self.loc_convs[result_conv_num](features))
            boxes_cls_scores.append(self.clf_convs[result_conv_num](features))
            result_conv_num += 1
        for i, (conv, loc_conv, clf_conv) in enumerate(zip(self.auxiliary_convs,
                                                           self.loc_convs[result_conv_num:],
                                                           self.clf_convs[result_conv_num:])):
            output = conv(output)
            if self.auxiliary_convs_appendices:
                features = self.auxiliary_convs_appendices[i](output)
            else:
                features = output
            boxes_gcx_gcy_gw_gh.append(loc_conv(features))
            boxes_cls_scores.append(clf_conv(features))
        return boxes_gcx_gcy_gw_gh, boxes_cls_scores


# PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=3, num_classes=3).to(DEVICE)
# print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])

# ===============================================================================

class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorAuxWithAppendix(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if norm_layer is None:
            norm_layer = functools.partial(
                nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError(
                "There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(
                    conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(
                    addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                output_channels = 6 * input_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

        prev_layer = self.backbone_convs[-1]
        for i in range(self._num_aux_layers):
            aux_conv = AuxiliaryConv(
                in_channels=prev_layer.out_channels,
                out_channels=prev_layer.out_channels,
                num_channels_decreasing_factor=aux_convs_decreasing_factor,
                bias=aux_convs_bias)
            input_channels = aux_conv.out_channels
            output_channels = 6 * input_channels
            appendix = [
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=1, bias=False)]
            if i != self._num_aux_layers - 1:
                appendix.append(nn.BatchNorm2d(
                    output_channels, eps=0.001, momentum=0.01))
            appendix.append(nn.ReLU())
            appendix = OutputConvAppendix(*appendix)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorAuxWithAppendix(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=3, num_classes=3).to(DEVICE)
# print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model

# ===============================================================================


# SuperDuper
class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(
                nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError(
                "There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(
                    conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(
                    addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels,
                              out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel,
                              out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels,
                              out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * \
                              self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                              out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel,
                              out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                              out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    #                     nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * \
                              self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED")
        for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices, ):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                # elif isinstance(m, (nn.ReLU, nn.Hardswish, nn.modules.activation.Ac))
                # else:
                #     raise TypeError(f"module of layer must be `nn.BatchNorm2d` or `nn.GroupNorm` instance, not {type(m)}")
        for layer in chain(self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
                # else:
                #     raise TypeError(f"module of layer must be `nn.Conv2d` instance, not {type(m)}")

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
# SuperDuper with dropout
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperWithDropout(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED")
        for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices, ):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                # elif isinstance(m, (nn.ReLU, nn.Hardswish, nn.modules.activation.Ac))
                # else:
                #     raise TypeError(f"module of layer must be `nn.BatchNorm2d` or `nn.GroupNorm` instance, not {type(m)}")
        for layer in chain(self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
                # else:
                #     raise TypeError(f"module of layer must be `nn.Conv2d` instance, not {type(m)}")

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
# SuperDuper No Dropout with Xavier on pred convs
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperNoDropoutXavierOnConvPreds(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    #                     nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED (XAVIER)")
        for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for layer in chain(self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    if m.bias is not None:
                        m.bias.zero_()

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
    # SuperDuper No Dropout with Xavier On all convs without gain
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperNoDropoutXavierAllConvsWoGain(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED (XAVIER ALL LAYERS WITHOUT GAIN)")
        for layer in chain(
                self.backbone_convs, self.auxiliary_convs,
                self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices,
                self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.zero_()
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
# Sup dup incorrect ident reproducing
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperReproducingIncorrentIdent(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    #                     nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )
        # incorrect ident

        @torch.no_grad()
        def _init(self):
            print("INIT CALLED")
            for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices, ):
                for m in layer.modules():
                    if m in self.potentially_trained_layers:
                        continue
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)
                    # elif isinstance(m, (nn.ReLU, nn.Hardswish, nn.modules.activation.Ac))
                    # else:
                    #     raise TypeError(f"module of layer must be `nn.BatchNorm2d` or `nn.GroupNorm` instance, not {type(m)}")
            for layer in chain(self.loc_convs, self.clf_convs):
                for m in layer.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.zeros_(m.bias)
                    # else:
                    #     raise TypeError(f"module of layer must be `nn.Conv2d` instance, not {type(m)}")

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
    # SuperDuper with kaiming_normal_ everywhere
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperNoDropoutWithKaiming_normal_Everywhere(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED (kaiming_normal_ everywhere)")
        for layer in chain(
                self.backbone_convs, self.auxiliary_convs,
                self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices,
                self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
    # SuperDuper no dropout with xavier on pred convs wo gain
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuperNoDropoutXavierOnPredConvsWithoutGAIN(ModifiedMobileNetV3SmallBackboneSSMBDLikeDetector):
    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=576, dropout=None):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_layers = list(self.backbone_convs)
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_small", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # 3-th layer needs larger receptive field so let's
            # build additional conv block as the next but without halving dimensions
            if layer_num == 3:
                conf = inverted_residual_setting[layer_num + 1]
                # increases receptive field without halving dimensions
                conf.stride = 1
                addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
                appendix.append(addit_conv_for_receptive_field)
                input_channels = addit_conv_for_receptive_field.out_channels
            else:
                input_channels = backbone[layer_num].out_channels

            # layer #12 is already such a conv
            if layer_num == 11:
                addit_conv_for_receptive_field = backbone[-1]
                self._potentially_trained_layers.append(addit_conv_for_receptive_field)
                appendix.append(addit_conv_for_receptive_field)
            else:
                # output_channels = 6 * input_channels
                output_channels = appendices_out_channels
                addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                    input_channels, output_channels, kernel_size=1,
                    norm_layer=norm_layer, activation_layer=nn.Hardswish)
                appendix.append(addit_conv_for_channels)

            appendix = OutputConvAppendix(*appendix)
            self.backbone_outputs_convs_appendices.append(appendix)

            # self.loc_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=appendix.out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            # output_channels = 6 * input_channels
            output_channels = appendices_out_channels
            addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                input_channels, output_channels, kernel_size=1,
                norm_layer=norm_layer, activation_layer=nn.Hardswish)
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(addit_conv_for_channels)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            # self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * self._num_classes,
            #                                 **pred_convs_kwargs))

            # self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
            #                                 out_channels=self._num_priors * 4,
            #                                 **pred_convs_kwargs))
            self.loc_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
                )
            )
            self.clf_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
                    nn.Hardswish(inplace=True),
                    # nn.Dropout(p=dropout, inplace=True),
                    nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
                )
            )

    @torch.no_grad()
    def _init(self):
        print("INIT CALLED (XAVIER WO GAIN on preds convs)")
        for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for layer in chain(self.loc_convs, self.clf_convs):
            for m in layer.modules():
                if m in self.potentially_trained_layers:
                    continue
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

# # PyTorch MobileNetV3-Small implementation pretrained on ImageNet
# net = torchvision.models.mobilenet_v3_small(pretrained=True)
# model = ModifiedMobileNetV3SmallBackboneSSMBDLikeDetectorSuperDuper(
#  net.features, feature_layers_nums=(3, 8, 11), num_aux_layers=2, num_classes=3).to(DEVICE)
# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
    # MobileNetV3Large
import torchvision.models.mobilenetv3 as mobilenetv3


class ModifiedMobileNetV3LargeBackboneSSMBDLikeDetectorSimpleConvs(SSMBDLikeDetector):
    def __init__(
            self,
            backbone: Iterable[nn.Module], feature_layers_nums: List[int],
            # bacbone_feature_layers_appendices: Iterable[nn.Module],
            num_aux_layers: int,
            num_classes: int, *,
            priors_aspect_ratios=(1, 2, 1 / 2, 3, 1 / 3),
            extra_priors_aspect_ratios=(1,),
            # start_scale=0.1,
            # end_scale=0.9
            norm_layer=None,
            aux_convs_decreasing_factor=1,
            aux_convs_bias=False,
            pred_convs_kwargs=None,
            all_preds_layers_have_same_input_size=False
    ):
        if pred_convs_kwargs is None:
            pred_convs_kwargs = dict(kernel_size=3, stride=1, padding=1)
        backbone = list(backbone)
        NUM_FEATURES_LAYERS = 17
        if len(backbone) != NUM_FEATURES_LAYERS:
            raise ValueError(f"`backbone` must contain exactly {NUM_FEATURES_LAYERS} layers")
        if not all([0 <= num < NUM_FEATURES_LAYERS - 1 for num in feature_layers_nums]):
            raise ValueError(f"`feature_layers_nums` must contain ints between 0 (incl) and {NUM_FEATURES_LAYERS - 1} (excl)")
        super().__init__(
            backbone, feature_layers_nums, num_aux_layers, num_classes,
            priors_aspect_ratios=priors_aspect_ratios, extra_priors_aspect_ratios=extra_priors_aspect_ratios, pred_convs_kwargs=pred_convs_kwargs,
            norm_layer=norm_layer, aux_convs_decreasing_factor=aux_convs_decreasing_factor, aux_convs_bias=aux_convs_bias, all_preds_layers_have_same_input_size=all_preds_layers_have_same_input_size)

    @property
    def potentially_trained_modules(self):
        return self._potentially_trained_modules

    def _predict_offsets_and_scores_impl(self, input):
        boxes_gcx_gcy_gw_gh = []
        boxes_cls_scores = []
        result_conv_num = 0
        output = input
        for conv in self.backbone_convs:
            output = conv(output)
            if conv not in self._backbone_feature_layers:
                continue
            if self.backbone_outputs_convs_appendices:
                output_appendix = self.backbone_outputs_convs_appendices[result_conv_num]
                features = output_appendix(output)
            else:
                features = output

            boxes_gcx_gcy_gw_gh.append(self.loc_convs[result_conv_num](features))
            boxes_cls_scores.append(self.clf_convs[result_conv_num](features))
            result_conv_num += 1
        for i, (conv, loc_conv, clf_conv) in enumerate(zip(self.auxiliary_convs,
                                                           self.loc_convs[result_conv_num:],
                                                           self.clf_convs[result_conv_num:])):
            output = conv(output)
            if self.auxiliary_convs_appendices:
                features = self.auxiliary_convs_appendices[i](output)
            else:
                features = output
            boxes_gcx_gcy_gw_gh.append(loc_conv(features))
            boxes_cls_scores.append(clf_conv(features))
        return boxes_gcx_gcy_gw_gh, boxes_cls_scores

    def _define_layers(self, backbone, pred_convs_kwargs, norm_layer, aux_convs_decreasing_factor, aux_convs_bias, appendices_out_channels=None, dropout=None, all_preds_layers_have_same_input_size=False):
        self.backbone_convs = nn.ModuleList(backbone[:-1])
        self._backbone_feature_layers = []
        self._potentially_trained_modules = list(chain.from_iterable(layer.modules() for layer in self.backbone_convs))
        self.backbone_outputs_convs_appendices = nn.ModuleList()
        self.auxiliary_convs = nn.ModuleList()
        self.auxiliary_convs_appendices = nn.ModuleList()
        self.loc_convs = nn.ModuleList()
        self.clf_convs = nn.ModuleList()

        if appendices_out_channels is None:
            appendices_out_channels = backbone[-1].out_channels

        if dropout is None:
            dropout = 0.2

        if norm_layer is None:
            norm_layer = functools.partial(nn.BatchNorm2d)
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            arch="mobilenet_v3_large", params={})

        # Pad setting list with None as the 1st layer of the backbone is not a residual block
        inverted_residual_setting = [None] + inverted_residual_setting
        # The last layer of backbone is not a residual either
        if len(inverted_residual_setting) != len(backbone) - 1:
            raise ValueError("There is mismatching between the passed backbone and residual settings")

        for layer_num in self._feature_layers_nums:
            layer = self.backbone_convs[layer_num]
            self._backbone_feature_layers.append(layer)

            appendix = []

            # # 3-th layer needs larger receptive field so let's
            # # build additional conv block as the next but without halving dimensions
            # if layer_num == 3:
            #     conf = inverted_residual_setting[layer_num + 1]
            #     # increases receptive field without halving dimensions
            #     conf.stride = 1
            #     addit_conv_for_receptive_field = mobilenetv3.InvertedResidual(conf, norm_layer)
            #     appendix.append(addit_conv_for_receptive_field)
            #     input_channels = addit_conv_for_receptive_field.out_channels
            # else:
            #     input_channels = backbone[layer_num].out_channels
            input_channels = backbone[layer_num].out_channels

            # layer #15 is already such a conv
            if layer_num == 15:
                addit_conv_for_channels = backbone[-1][0]
                self._potentially_trained_modules.extend(addit_conv_for_channels.modules())
                appendix.append(addit_conv_for_channels)
                appendix.append(nn.Hardswish())
                # output_channels = addit_conv_for_channels

            else:
                if all_preds_layers_have_same_input_size:
                    output_channels = appendices_out_channels
                else:
                    output_channels = 6 * input_channels
                # addit_conv_for_channels = mobilenetv3.ConvBNActivation(
                #     input_channels, output_channels, kernel_size=1,
                #     norm_layer=norm_layer, activation_layer=nn.Hardswish)
                addit_conv_for_channels = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
                appendix.append(addit_conv_for_channels)
                appendix.append(nn.Hardswish())
            appendix = OutputConvAppendix(*appendix)
            appendix_out_channels = appendix.out_channels
            # appendix_out_channels = output_channels
            self.backbone_outputs_convs_appendices.append(appendix)

            self.loc_convs.append(nn.Conv2d(in_channels=appendix_out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=appendix_out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))
            # self.loc_convs.append(
            #     nn.Sequential(
            #         nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
            #         nn.Hardswish(inplace=True),
            #         # nn.Dropout(p=dropout, inplace=True),
            #         nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
            #     )
            # )
            # self.clf_convs.append(
            #     nn.Sequential(
            #         nn.Conv2d(in_channels=appendix.out_channels, out_channels=last_channel, kernel_size=1),
            #         nn.Hardswish(inplace=True),
            #         # nn.Dropout(p=dropout, inplace=True),
            #         nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
            #     )
            # )

        # prev_layer = self.backbone_convs[-1]
        conf = inverted_residual_setting[-1]
        # conf.stride = 1  # cannot use it, because it is impossible to pass padding to Residual constructor
        conf.stride = 2
        conf.kernel = 5  # by default,
        # conf.kernel = 3 # or use can reduce it
        for i in range(self._num_aux_layers):
            # aux_conv = AuxiliaryConv(
            #     in_channels=prev_layer.out_channels,
            #     out_channels=prev_layer.out_channels,
            #     num_channels_decreasing_factor=aux_convs_decreasing_factor,
            #     bias=aux_convs_bias)
            aux_conv = mobilenetv3.InvertedResidual(conf, norm_layer)
            input_channels = aux_conv.out_channels
            if all_preds_layers_have_same_input_size:
                output_channels = appendices_out_channels
            else:
                output_channels = 6 * input_channels
            # addit_conv_for_channels = mobilenetv3.ConvBNActivation(
            #     input_channels, output_channels, kernel_size=1,
            #     norm_layer=norm_layer, activation_layer=nn.Hardswish)
            appendix = []
            addit_conv_for_channels = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
            appendix.append(addit_conv_for_channels)
            appendix.append(nn.Hardswish())
            # appendix = [
            #     nn.Conv2d(
            #         in_channels=input_channels,
            #         out_channels=output_channels,
            #         kernel_size=1, bias=False)]
            # if i != self._num_aux_layers - 1:
            #     appendix.append(nn.BatchNorm2d(output_channels, eps=0.001, momentum=0.01))
            # appendix.append(nn.ReLU())
            # appendix = OutputConvAppendix(*appendix)
            appendix = OutputConvAppendix(*appendix)
            self.auxiliary_convs_appendices.append(appendix)
            aux_conv_branch_out_channels = output_channels
            # prev_layer = aux_conv
            self.auxiliary_convs.append(aux_conv)
            self.loc_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                                            out_channels=self._num_priors * 4,
                                            **pred_convs_kwargs))
            self.clf_convs.append(nn.Conv2d(in_channels=aux_conv_branch_out_channels,
                                            out_channels=self._num_priors * self._num_classes,
                                            **pred_convs_kwargs))
            # self.loc_convs.append(
            #     nn.Sequential(
            #         nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
            #         nn.Hardswish(inplace=True),
            #         # nn.Dropout(p=dropout, inplace=True),
            #         nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * 4, kernel_size=1),
            #     )
            # )
            # self.clf_convs.append(
            #     nn.Sequential(
            #         nn.Conv2d(in_channels=aux_conv_branch_out_channels, out_channels=last_channel, kernel_size=1),
            #         nn.Hardswish(inplace=True),
            #         # nn.Dropout(p=dropout, inplace=True),
            #         nn.Conv2d(in_channels=last_channel, out_channels=self._num_priors * self._num_classes, kernel_size=1),
            #     )
            # )

#     @torch.no_grad()
#     def _init(self):
#         print("INIT CALLED (XAVIER WO GAIN on preds convs)")
#         for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices):
#             for m in layer.modules():
#                 if m in self.potentially_trained_modules:
#                     continue
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
#         for layer in chain(self.loc_convs, self.clf_convs):
#             for m in layer.modules():
#                 if m in self.potentially_trained_modules:
#                     continue
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#     @torch.no_grad()
#     def _init(self):
#         print("INIT CALLED (INIT_kaiming_n_usual_norm_on_preds)")
#         for layer in chain(self.backbone_convs, self.auxiliary_convs, self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices):
#             for m in layer.modules():
#                 if m in self.potentially_trained_modules:
#                     continue
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
#         for layer in chain(self.loc_convs, self.clf_convs):
#             for m in layer.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.normal_(m.weight, 0, 0.01)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#     @torch.no_grad()
#     def _init(self):
#         print("INIT CALLED (Kaiming normal on ConvBNA, Xavier uniform on simple convs (bias zeroed))")
#         for layer in chain(self.backbone_convs, self.auxiliary_convs):
#             for m in layer.modules():
#                 if m in self.potentially_trained_modules:
#                     continue
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out')
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                     nn.init.ones_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#         for layer in chain(self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices, self.loc_convs, self.clf_convs):
#             for m in layer.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#     @torch.no_grad()
#     def _init(self):
#         # _init() from
#         print("INIT CALLED (Xavier_uniform_everywhere)")
#         # original = next(self.backbone_convs[0].parameters()).clone()
#         for layer in chain(
#                 self.backbone_convs, self.auxiliary_convs,
#                 self.backbone_outputs_convs_appendices, self.auxiliary_convs_appendices,
#                 self.loc_convs, self.clf_convs):
#             for m in layer.modules():
#                 if m in self.potentially_trained_modules:
#                     continue
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.zeros_(m.bias)
#                 elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                     nn.init.ones_(m.weight)
#                     nn.init.zeros_(m.bias)
#         # after = next(self.backbone_convs[0].parameters()).clone()
#         # assert torch.allclose(original, after), torch.allclose(original, after)
    @torch.no_grad()
    def _init(self):
        print("INIT CALLED (no init)")


# PyTorch MobileNetV3-Large implementation pretrained on ImageNet
# net_large = torchvision.models.mobilenet_v3_large(pretrained=True)
# model = ModifiedMobileNetV3LargeBackboneSSMBDLikeDetectorSimpleConvs(
#     net_large.features, feature_layers_nums=(6, 12, 15), num_aux_layers=2, num_classes=3).to(DEVICE)

# # print(model)
# batch = torch.randn(3, 224, 224).unsqueeze(0)
# results = model(batch)
# print([r.shape for r in results])
# model
# ===============================================================================
# for one box
class SimpleCNNExactlyOneBoxDetector(nn.Module):
    def __init__(self):
        super().__init__()
        ...
# ===============================================================================
