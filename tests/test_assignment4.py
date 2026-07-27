import math
import pathlib
import sys

import torch
from torch import nn

A4_DIR = pathlib.Path(__file__).resolve().parents[1] / "assignments" / "a4-object-detection"
sys.path.insert(0, str(A4_DIR))

import two_stage_detector as tsd
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


def test_faster_rcnn_roi_logits_and_targets_share_level_major_order(monkeypatch):
    """RoIAlign groups boxes by FPN level, then by image within each level."""

    class FakeRPN(nn.Module):
        def forward(self, feats, strides, gt_boxes=None):
            # x1 encodes the expected shifted class ID (1, 2, or 3). Boxes of
            # different classes use separate y ranges so matching is unambiguous.
            boxes = {
                1: torch.tensor([1.0, 1.0, 5.0, 5.0]),
                2: torch.tensor([2.0, 10.0, 6.0, 14.0]),
                3: torch.tensor([3.0, 20.0, 7.0, 24.0]),
            }
            device = feats["p3"].device
            proposals = {
                "p3": [boxes[1][None].to(device), boxes[2][None].to(device)],
                "p4": [boxes[3][None].to(device), boxes[1][None].to(device)],
                "p5": [boxes[2][None].to(device), boxes[3][None].to(device)],
            }
            zero = feats["p3"].sum() * 0.0
            return {
                "proposals": proposals,
                "loss_rpn_obj": zero,
                "loss_rpn_box": zero,
            }

    class CodedHead(nn.Module):
        def forward(self, roi_feats):
            shifted_classes = roi_feats[:, 0, 0, 0].round().long()
            logits = roi_feats.new_full((len(shifted_classes), 4), -10.0)
            logits[torch.arange(len(logits)), shifted_classes] = 10.0
            return logits

    def fake_roi_align(feats, boxes, output_size, spatial_scale, aligned):
        codes = torch.cat([box[:, 0] for box in boxes])
        return codes[:, None, None, None].expand(
            -1, feats.shape[1], output_size[0], output_size[1]
        )

    def take_all_foreground(matched_boxes, num_samples, fg_fraction):
        return (
            torch.arange(len(matched_boxes), device=matched_boxes.device),
            torch.empty(0, dtype=torch.long, device=matched_boxes.device),
        )

    monkeypatch.setattr(tsd, "mix_gt_with_proposals", lambda proposals, gt: proposals)
    monkeypatch.setattr(tsd.torchvision.ops, "roi_align", fake_roi_align)
    monkeypatch.setattr(tsd, "sample_rpn_training", take_all_foreground)

    model = FasterRCNN(
        FakeBackbone(8), FakeRPN(), stem_channels=[8], num_classes=3,
        batch_size_per_image=8,
    )
    model.cls_pred = CodedHead()
    model.train()

    class_boxes = [
        [1.0, 1.0, 5.0, 5.0, 0.0],
        [2.0, 10.0, 6.0, 14.0, 1.0],
        [3.0, 20.0, 7.0, 24.0, 2.0],
    ]
    gt_boxes = torch.tensor([class_boxes, class_boxes])
    images = torch.zeros(2, 3, 32, 32)

    losses = model(images, gt_boxes)
    assert losses["loss_cls"] < 1e-6


def test_voc_validation_resize_does_not_center_crop(tmp_path):
    from torchvision import transforms
    from a4_helper import VOC2007DetectionTiny

    (tmp_path / "voc07_train.json").write_text("[]")
    (tmp_path / "voc07_val.json").write_text("[]")
    train = VOC2007DetectionTiny(str(tmp_path), split="train", image_size=224)
    val = VOC2007DetectionTiny(str(tmp_path), split="val", image_size=224)

    assert any(isinstance(op, transforms.CenterCrop) for op in train.image_transform.transforms)
    assert not any(isinstance(op, transforms.CenterCrop) for op in val.image_transform.transforms)


def test_inference_writes_gt_even_when_detector_has_no_predictions(tmp_path):
    from a4_helper import inference_with_detector

    class EmptyDetector(nn.Module):
        def forward(
            self, images, gt_boxes=None, test_score_thresh=None, test_nms_thresh=None
        ):
            return (
                torch.empty((0, 4), device=images.device),
                torch.empty((0,), dtype=torch.long, device=images.device),
                torch.empty((0,), device=images.device),
            )

    gt_boxes = torch.tensor(
        [[[1.0, 2.0, 6.0, 7.0, 0.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]]
    )
    loader = [(["example.jpg"], torch.zeros(1, 3, 8, 8), gt_boxes)]
    output_dir = tmp_path / "map-input"

    inference_with_detector(
        EmptyDetector(),
        loader,
        {0: "object"},
        score_thresh=0.2,
        nms_thresh=0.5,
        output_dir=str(output_dir),
        device="cpu",
    )

    gt_path = output_dir / "ground-truth" / "example.txt"
    det_path = output_dir / "detection-results" / "example.txt"
    assert gt_path.read_text().strip() == "object 1.00 2.00 6.00 7.00"
    assert det_path.exists()
    assert det_path.read_text() == ""
