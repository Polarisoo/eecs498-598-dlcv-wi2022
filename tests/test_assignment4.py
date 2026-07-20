import math
import pathlib
import sys

import torch
from torch import nn

A4_DIR = pathlib.Path(__file__).resolve().parents[1] / "assignments" / "a4-object-detection"
sys.path.insert(0, str(A4_DIR))

import one_stage_detector as osd
from common import get_fpn_location_coords, nms
from one_stage_detector import (
    FCOS,
    FCOSPredictionNetwork,
    fcos_apply_deltas_to_locations,
    fcos_get_deltas_from_locations,
    fcos_make_centerness_targets,
)
from two_stage_detector import (
    FasterRCNN,
    RPN,
    RPNPredictionNetwork,
    generate_fpn_anchors,
    iou,
    rcnn_apply_deltas_to_anchors,
    rcnn_get_deltas_from_anchors,
)


class FakeBackbone(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.out_channels = out_channels
        self.fpn_strides = {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images):
        b, _, h, w = images.shape
        device, dtype = images.device, images.dtype
        return {
            "p3": torch.zeros(b, self.out_channels, h // 8, w // 8, device=device, dtype=dtype),
            "p4": torch.zeros(b, self.out_channels, h // 16, w // 16, device=device, dtype=dtype),
            "p5": torch.zeros(b, self.out_channels, h // 32, w // 32, device=device, dtype=dtype),
        }


def test_common_locations_and_nms():
    locs = get_fpn_location_coords({"p3": (1, 8, 2, 3)}, {"p3": 8})["p3"]
    expected = torch.tensor(
        [[4.0, 4.0], [12.0, 4.0], [20.0, 4.0], [4.0, 12.0], [12.0, 12.0], [20.0, 12.0]]
    )
    assert torch.allclose(locs, expected)

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0], [20.0, 20.0, 30.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_fcos_heads_and_box_math():
    torch.manual_seed(0)
    net = FCOSPredictionNetwork(4, 8, [16, 16])
    feats = {"p3": torch.randn(2, 8, 4, 5), "p4": torch.randn(2, 8, 2, 3), "p5": torch.randn(2, 8, 1, 2)}
    cls_logits, box_deltas, ctr_logits = net(feats)
    assert cls_logits["p3"].shape == (2, 20, 4)
    assert box_deltas["p3"].shape == (2, 20, 4)
    assert ctr_logits["p3"].shape == (2, 20, 1)

    locs = torch.tensor([[8.0, 8.0], [4.0, 4.0]])
    gt = torch.tensor([[0.0, 0.0, 16.0, 16.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])
    deltas = fcos_get_deltas_from_locations(locs, gt, 8)
    assert torch.allclose(deltas[0], torch.ones(4))
    assert torch.all(deltas[1] == -1)
    assert torch.allclose(fcos_apply_deltas_to_locations(deltas[:1], locs[:1], 8), gt[:1, :4])
    assert torch.allclose(fcos_make_centerness_targets(deltas), torch.tensor([1.0, -1.0]))


def test_rcnn_anchor_iou_and_delta_math():
    locs = get_fpn_location_coords({"p3": (1, 8, 2, 2)}, {"p3": 8})["p3"]
    anchors = generate_fpn_anchors({"p3": locs}, {"p3": 8}, 4)["p3"]
    assert anchors.shape == (12, 4)

    overlaps = iou(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), torch.tensor([[5.0, 5.0, 15.0, 15.0], [0.0, 0.0, 10.0, 10.0]]))
    assert torch.allclose(overlaps, torch.tensor([[1.0 / 7.0, 1.0]]), atol=1e-6)

    anchor = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    deltas = rcnn_get_deltas_from_anchors(anchor, gt)
    assert torch.allclose(deltas, torch.tensor([[0.5, 0.5, math.log(2.0), math.log(2.0)]]), atol=1e-6)
    assert torch.allclose(rcnn_apply_deltas_to_anchors(deltas.clone(), anchor), gt, atol=1e-5)


def test_rpn_prediction_and_training_forward():
    torch.manual_seed(0)
    feats = {"p3": torch.randn(2, 8, 4, 4), "p4": torch.randn(2, 8, 2, 2), "p5": torch.randn(2, 8, 1, 1)}
    pred = RPNPredictionNetwork(8, [8], 3)
    obj_logits, box_deltas = pred(feats)
    assert obj_logits["p3"].shape == (2, 48)
    assert box_deltas["p3"].shape == (2, 48, 4)

    gt_boxes = torch.tensor(
        [
            [[4.0, 4.0, 20.0, 20.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            [[8.0, 8.0, 24.0, 24.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
        ]
    )
    rpn = RPN(8, [8], batch_size_per_image=16, anchor_stride_scale=2, pre_nms_topk=16, post_nms_topk=8)
    rpn.train()
    out = rpn(feats, {"p3": 8, "p4": 16, "p5": 32}, gt_boxes)
    assert set(out) == {"proposals", "loss_rpn_obj", "loss_rpn_box"}
    assert torch.isfinite(out["loss_rpn_obj"])
    assert torch.isfinite(out["loss_rpn_box"])


def test_fcos_forward_with_fake_backbone(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(osd, "DetectorBackboneWithFPN", FakeBackbone)
    model = FCOS(num_classes=3, fpn_channels=8, stem_channels=[8])
    images = torch.zeros(2, 3, 32, 32)
    gt_boxes = torch.tensor(
        [
            [[4.0, 4.0, 20.0, 20.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            [[8.0, 8.0, 24.0, 24.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
        ]
    )
    model.train()
    losses = model(images, gt_boxes)
    assert set(losses) == {"loss_cls", "loss_box", "loss_ctr"}
    assert all(torch.isfinite(v) for v in losses.values())

    model.eval()
    boxes, classes, scores = model(images[:1], test_score_thresh=0.0, test_nms_thresh=0.5)
    assert boxes.shape[1] == 4
    assert classes.ndim == 1
    assert scores.ndim == 1


def test_faster_rcnn_forward_with_fake_backbone():
    torch.manual_seed(0)
    backbone = FakeBackbone(8)
    rpn = RPN(8, [8], batch_size_per_image=16, anchor_stride_scale=2, pre_nms_topk=16, post_nms_topk=8)
    model = FasterRCNN(backbone, rpn, stem_channels=[8], num_classes=3, batch_size_per_image=8)
    images = torch.zeros(2, 3, 32, 32)
    gt_boxes = torch.tensor(
        [
            [[4.0, 4.0, 20.0, 20.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
            [[8.0, 8.0, 24.0, 24.0, 2.0], [-1.0, -1.0, -1.0, -1.0, -1.0]],
        ]
    )
    model.train()
    losses = model(images, gt_boxes)
    assert set(losses) == {"loss_rpn_obj", "loss_rpn_box", "loss_cls"}
    assert all(torch.isfinite(v) for v in losses.values())

    model.eval()
    boxes, classes, scores = model(images[:1], test_score_thresh=0.0, test_nms_thresh=0.5)
    assert boxes.shape[1] == 4
    assert classes.ndim == 1
    assert scores.ndim == 1
