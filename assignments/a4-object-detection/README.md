# Assignment 4: Object Detection

Official assignment page: https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment4.html

This directory contains the official A4 starter layout plus completed Python implementations for the object-detection exercises. The notebooks are preserved for JupyterLab or Colab use.

## Implemented files

- `common.py`: FPN lateral/output layers, FPN forward pass, FPN location coordinates, non-maximum suppression.
- `one_stage_detector.py`: FCOS prediction head, location/box delta transforms, centerness targets, training losses, and inference post-processing.
- `two_stage_detector.py`: RPN prediction head, FPN anchors, IoU, Faster R-CNN box transforms, RPN training/proposal logic, RoI Align classifier head, second-stage loss, and inference.
- `eecs598/utils.py`: OpenCV is imported lazily inside the attention visualizer so core tests run on Windows environments where `cv2` DLL loading fails.

## Local JupyterLab notes

Start JupyterLab from this directory and skip the Google Drive / Colab setup cells when working locally:

```powershell
cd assignments/a4-object-detection
jupyter lab
```

The notebooks still contain the official training and submission cells. For a real course submission, run the required training cells with a GPU runtime and fill in the requested student metadata.

## Validation

From the repository root:

```powershell
python -m pytest tests/test_assignment4.py -q
python -m pytest -q
```

Current local validation: `6 passed` for A4 and `40 passed` for the full repository.

## Artifact caveat

The official A4 submission expects trained detector checkpoints such as `fcos_detector.pt` and `rcnn_detector.pt`. These GPU training artifacts were not generated in this CPU-oriented Codex pass, and `.pt` files are intentionally excluded from Git.
