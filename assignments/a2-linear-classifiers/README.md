# Assignment 2: Linear Classifiers and Two-Layer Networks

Official instructions: [Assignment 2](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment2.html)

## Files

- `linear_classifier.ipynb`: SVM and Softmax walkthrough and visible checks
- `linear_classifier.py`: naive/vectorized losses, SGD, prediction, tuning
- `two_layer_net.ipynb`: two-layer fully-connected network walkthrough
- `two_layer_net.py`: forward/backward passes, training, prediction, tuning
- `challenge_problem.ipynb`: manually designed MNIST network challenge
- `eecs598/`: course-provided data, grading, and visualization utilities

## Run locally with JupyterLab

Start `jupyter lab` from this directory and skip the **Google Colab Setup**
cells. Most functions can be developed on CPU by passing `device="cpu"` to toy
helpers and model constructors. The official CIFAR-10 cells default to CUDA and
should be run in the dedicated CUDA environment or a Colab GPU runtime.

Implementations belong only inside the starter's `TODO` regions. The notebooks
remain in their official starter form so students can execute and annotate them
in their own environment.

## Validation status

The data-free repository tests check naive/vectorized agreement, numerical
stability, analytic gradients, SGD updates, prediction, and toy-data training.
Full CIFAR-10 tuning, saved best-model checkpoints, notebook outputs, and the
manual MNIST challenge still require an interactive GPU/notebook run.
