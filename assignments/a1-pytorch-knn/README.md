# Assignment 1: PyTorch 101 and k-Nearest Neighbors

Official instructions: [Assignment 1](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment1.html)

## Files

- `pytorch101.ipynb`: PyTorch introduction and visible checks
- `pytorch101.py`: implementations for the tensor programming tasks
- `knn.ipynb`: CIFAR-10 / kNN experiment and visible checks
- `knn.py`: distance functions, label prediction, classifier, cross-validation
- `eecs598/`: course-provided utilities

## Run locally with JupyterLab

Start `jupyter lab` from this directory. The official notebooks explicitly allow local users to skip **Google Colab Setup**, so do not execute the Google Drive mounting cells. On Windows, also skip the Colab path-check cell containing `time.tzset()`.

Implementations live inside the `BEGIN/END OF YOUR CODE` regions in the Python files. Do not alter notebook cell structure. Restart the kernel and rerun checks if autoreload appears to retain stale imports.

## GPU and memory notes

- `mm_on_gpu` follows the prompt and requires CUDA; use a Colab GPU runtime on a CPU-only computer.
- The `10000 x 10000` GPU demonstration and full CIFAR-10 kNN cells use a lot of memory. Run small-sample checks first.
- CIFAR-10 is downloaded on first use and is excluded by `.gitignore`.

## Before Autograder submission

1. Enter your name and UMID at the top of both notebooks.
2. Run the required notebook cells in Colab and keep their outputs.
3. Remove any temporary cells.
4. Package only `pytorch101.py`, `pytorch101.ipynb`, `knn.py`, and `knn.ipynb`.

The starter's `eecs598/submit.py` may raise `NameError` in its no-argument user-info branch. If needed, pass `uniquename` and `umid` explicitly or make the four-file archive manually.
