# EECS 498-007 / 598-005: Deep Learning for Computer Vision (Winter 2022)

This public study repository maintains programming assignments for the University of Michigan Winter 2022 course.

- [Course home](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)
- [Schedule](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)
- [Assignment 1](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment1.html)
- [Assignment 2](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment2.html)
- [Assignment 3](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html)
- [Assignment 4](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment4.html)
- [Assignment 5](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html)
- [Assignment 6](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment6.html)

> Course submissions must be each student's own work. The syllabus prohibits looking at or distributing solutions to similar assignments. This repository is public at the user's request, so do not treat it as an official course-submission channel or reuse the code in a class where public solution sharing is prohibited. This repository has no open-source license.

## Progress

Progress snapshot: **2026-07-27**. For this progress report, "complete" means
the required implementation and core experiments are complete; final course
submission archives and administrative packaging cells are tracked separately.

| Assignment | Topic | Status |
| --- | --- | --- |
| A1 | PyTorch 101; k-Nearest Neighbors | **Complete** - required PyTorch and kNN implementations are present and locally tested |
| A2 | Linear classifiers; two-layer network | **Complete** - all three notebooks executed without notebook errors (13/13, 34/34, and 24/24 code cells) |
| A3 | Modular API; CNN; BatchNorm; Autograd | **Complete** - both core notebooks executed without errors; only the final submission-packaging cell is not preserved |
| A4 | Object detection | **In progress** - FCOS/one-stage training is complete (validation mAP about **25.47%**); corrected Faster R-CNN/two-stage training is still running locally and needs final mAP validation |
| A5 | RNNs; image captioning; Transformers | **In progress** - Colab compatibility issues are fixed and RNN captioning is training; LSTM, attention captioning, Transformers, and saved final outputs remain |
| A6 | VAE; GAN; style transfer; visualization | **Complete** - all four notebooks executed with saved outputs and no notebook errors (23/23, 15/15, 23/23, and 15/15 code cells) |

### Work remaining for A4 and A5

- **A4 Two-stage detector:** the first completed Faster R-CNN run produced an
  invalid 0.60% mAP because RoI features and matched labels used inconsistent
  batch ordering. That issue has been fixed. A clean 9,000-iteration local GPU
  retraining run is in progress; the remaining work is to save the corrected
  checkpoint, run validation, and record the final mAP.
- **A5 RNN/LSTM captioning and Transformers:** the active Colab notebook now
  has corrected repository-path setup, a current Matplotlib style name, and a
  compatible ImageEncoder call. RNN training is underway. The remaining work
  is to finish RNN, LSTM, and attention-captioning training, run
  Transformers.ipynb, and save the final notebook outputs back to GitHub.

## Recent updates

As of 2026-07-27:

- A2-A5 notebooks were adapted for the Colab workflow that clones this GitHub
  repository directly, instead of relying on a manually mounted Google Drive
  folder.
- A2 now includes hand_drawn_weights.jpeg for the challenge problem.
- A2, A3, and all four A6 notebooks have been executed and checked for notebook
  error outputs.
- A4 FCOS training and evaluation are complete; corrected Faster R-CNN
  retraining is underway.
- A5 Colab execution has started after resolving current-runtime compatibility
  issues; completed outputs are not yet committed.

## Repository layout

```text
assignments/
|-- a1-pytorch-knn/
|-- a2-linear-classifiers/
|-- a3-neural-networks/
|-- a4-object-detection/
|-- a5-rnn-transformers/
`-- a6-generative-models/
```

Each assignment preserves the official starter's relative layout. Datasets, caches, checkpoints, and submission archives are excluded from Git.

## JupyterLab setup

Python 3.11 is recommended. The official notebooks are Colab-first: local JupyterLab is useful for editing and CPU tests, while GPU cells are best run in a Colab GPU runtime.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name eecs498-wi2022 --display-name "Python (eecs498-wi2022)"
cd assignments\a1-pytorch-knn
jupyter lab
```

When opening an official notebook locally, skip the **Google Colab Setup** mounting cells and start JupyterLab from the assignment directory. Before a course submission, fill in your name and UMID, run the required cells in Colab, and retain their outputs.

## Validation

Run from the repository root:

```powershell
python -m pytest
```

CPU tests cover A1 tensor exercises and kNN; A2 SVM/Softmax losses, gradients,
SGD, and two-layer-network passes; A3 modular layers, optimizers, dropout,
convolution, pooling, BatchNorm, and complete-network integration; A4 FPN
utilities, FCOS, RPN, Faster R-CNN box math, losses, and fake-backbone forward
passes; and A5 RNN/LSTM/attention captioning plus Transformer blocks and forward
passes. CUDA-only tests are skipped when no NVIDIA GPU is available.

There is not yet a dedicated `tests/test_assignment6.py`; A6 should be verified
by running the four A6 notebooks in a Colab GPU runtime and keeping the required
outputs in the notebooks. Accuracy, generated samples, visualizations, and final
checkpoint cells should still be run from the course notebooks.

## Remaining work

- Finish and validate the corrected A4 two-stage detector, then preserve its
  final mAP output.
- Finish A5 RNN/LSTM/attention captioning and Transformers.ipynb, then save
  the completed Colab outputs back to the repository.
- Add a lightweight A6 pytest smoke-test suite if this repository should track
  A6 implementation correctness locally, the same way it currently tracks
  A1-A5.
- Keep datasets, checkpoints, caches, and submission ZIP files outside Git.
