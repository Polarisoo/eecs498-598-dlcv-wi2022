# EECS 498-007 / 598-005: Deep Learning for Computer Vision (Winter 2022)

This public study repository maintains programming assignments for the University of Michigan Winter 2022 course.

- [Course home](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/)
- [Schedule](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html)
- [Assignment 1](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment1.html)
- [Assignment 2](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment2.html)
- [Assignment 3](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html)
- [Assignment 4](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment4.html)

> Course submissions must be each student's own work. The syllabus prohibits looking at or distributing solutions to similar assignments. This repository is public at the user's request, so do not treat it as an official course-submission channel or reuse the code in a class where public solution sharing is prohibited. This repository has no open-source license.

## Progress

| Assignment | Topic | Status |
| --- | --- | --- |
| A1 | PyTorch 101; k-Nearest Neighbors | Implemented and locally tested |
| A2 | Linear classifiers; two-layer network | Core implementation complete; GPU experiments pending |
| A3 | Modular API; CNN; BatchNorm; Autograd | Core implementation complete; GPU experiments pending |
| A4 | Object detection | Core implementation complete; GPU training artifacts pending |
| A5 | RNNs; image captioning; Transformers | Not started |
| A6 | VAE; GAN; style transfer; visualization | Not started |

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
convolution, pooling, BatchNorm, and complete-network integration; and A4 FPN
utilities, FCOS, RPN, Faster R-CNN box math, losses, and fake-backbone forward
passes. CUDA-only tests are skipped when no NVIDIA GPU is available. Accuracy
and final checkpoint cells should still be run from the course notebooks.
