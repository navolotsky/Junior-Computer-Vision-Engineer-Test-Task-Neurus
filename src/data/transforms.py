import torch
import torchvision
from torchvision.ops import box_convert

from .. import BOX_FORMAT


class ConvertBoundingBoxesToRelativeForm(torch.nn.Module):
    def __init__(self, boundig_boxes_in_fmt: BOX_FORMAT = 'xyxy'):
        super().__init__()
        self._bbs_in_fmt = boundig_boxes_in_fmt

    def forward(self, image: torch.Tensor, boundig_boxes: torch.Tensor):
        *_, img_h, img_w = image.shape
        if self._bbs_in_fmt != 'xyxy':
            boundig_boxes = box_convert(
                boundig_boxes, in_fmt=self._bbs_in_fmt, out_fmt='xyxy')
        boundig_boxes = boundig_boxes / \
            torch.tensor([img_w, img_h, img_w, img_h])
        if self._bbs_in_fmt != 'xyxy':
            boundig_boxes = box_convert(
                boundig_boxes, in_fmt='xyxy', out_fmt=self._bbs_in_fmt)
        return image, boundig_boxes


class ResizeWithBoundingBoxes(torchvision.transforms.Resize):
    def forward(self, image: torch.Tensor, boundig_boxes: torch.Tensor):
        resized_image = super().forward(image)
        ratio_in_whwh_form = torch.div(
            torch.as_tensor(resized_image.shape[-2:]),
            torch.as_tensor(image.shape[-2:])
        ).flip(0).repeat(2)
        resized_bounding_boxes = boundig_boxes * ratio_in_whwh_form
        return resized_image, resized_bounding_boxes.type(boundig_boxes.dtype)


class ComposeWithUsingBoundingBoxes(torchvision.transforms.Compose):
    _default_transforms_applicable_to_boxes = [
        ResizeWithBoundingBoxes,
        ConvertBoundingBoxesToRelativeForm
    ]

    def __init__(
            self, transforms,
            transforms_applicable_to_bounding_boxes=None,
            override_default_transforms_applicable_to_boxes=False):
        super().__init__(transforms)
        self.transforms_applicable_to_boxes = (
            [] if override_default_transforms_applicable_to_boxes
            else self._default_transforms_applicable_to_boxes.copy()
        )
        if transforms_applicable_to_bounding_boxes is not None:
            self.transforms_applicable_to_boxes += list(
                transforms_applicable_to_bounding_boxes)

    def _single_transform_call_impl(self, transform, img, boxes):
        if (
            transform in self.transforms_applicable_to_boxes or
            type(transform) in self.transforms_applicable_to_boxes
        ):
            img, boxes = transform(img, boxes)
        else:
            img = transform(img)
        return img, boxes

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = self._single_transform_call_impl(t, img, boxes)
        return img, boxes
