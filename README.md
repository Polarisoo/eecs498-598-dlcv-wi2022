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

| Assignment | Topic | Status |
| --- | --- | --- |
| A1 | PyTorch 101; k-Nearest Neighbors | Implemented and locally tested |
| A2 | Linear classifiers; two-layer network | Core implementation complete; GPU experiments pending |
| A3 | Modular API; CNN; BatchNorm; Autograd | Core implementation complete; GPU experiments pending |
| A4 | Object detection | Core implementation complete; GPU training artifacts pending |
| A5 | RNNs; image captioning; Transformers | Core implementation complete; GPU training artifacts pending |
| A6 | VAE; GAN; style transfer; visualization | Code and notebooks added; Colab/GPU validation pending |

## Recent updates

As of 2026-07-27, `origin/main` includes a batch of updates after commit
`570384f`:

- A2-A5 notebooks were adapted for the Colab workflow that clones this GitHub
  repository directly, instead of relying on a manually mounted Google Drive
  folder.
- A2 now includes `hand_drawn_weights.jpeg` for the challenge problem.
- A4 includes a small dataset-download helper update in `a4_helper.py`.
- A6 has been added under `assignments/a6-generative-models/`, including VAE,
  GAN, network-visualization, and style-transfer notebooks, their companion
  `.py` implementation files, course utility modules, and style-transfer image
  assets.

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

- Run and preserve the official GPU notebook outputs for A2-A6 before treating
  the notebooks as final course-submission artifacts.
- Add a lightweight A6 pytest smoke-test suite if this repository should track
  A6 implementation correctness locally, the same way it currently tracks
  A1-A5.
- Keep datasets, checkpoints, caches, and submission ZIP files outside Git.
