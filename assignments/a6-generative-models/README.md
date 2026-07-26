# Assignment 6: Generative Models and Network Visualization

Official assignment page: https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment6.html

This directory contains the official A6 starter layout with completed implementations for all four exercises.

## Files

- `variational_autoencoders.ipynb` + `vae.py`: FC-VAE and conditional VAE on MNIST (encoder/decoder, reparametrization trick, ELBO loss)
- `generative_adversarial_networks.ipynb` + `gan.py`: vanilla GAN, LS-GAN, and DCGAN on MNIST (models, BCE/least-squares losses, optimizers)
- `network_visualization.ipynb` + `network_visualization.py`: saliency maps, adversarial attacks, and class visualization with a pretrained SqueezeNet
- `style_transfer.ipynb` + `style_transfer.py`: content/style/TV losses, Gram matrices, and guided style transfer
- `a6_helper.py`, `eecs598/`, `images/`: course-provided utilities and image assets

## Running in Colab

The Google Colab Setup cells were adapted for this repository: instead of
mounting Google Drive, they `git clone` this repo into the runtime and point
`GOOGLE_DRIVE_PATH` at `assignments/a6-generative-models`. Open any of the four
notebooks in Colab, switch to a GPU runtime (Runtime -> Change runtime type),
and "Run all" — a GPU is required (`a6_helper.train_vae` and the notebook
training loops use CUDA devices). MNIST, the ImageNet validation subset, and
the style-transfer data are downloaded at runtime and are excluded from Git.

## Compatibility notes for modern environments

Small fixes were applied to the 2022 starter so it runs on current Colab
(PyTorch >= 2.6 / torchvision >= 0.21 / SciPy >= 1.12 / Python >= 3.10):

- `scipy.ndimage.filters` import replaced with `scipy.ndimage` in `a6_helper.py`
- `DataLoader.__iter__().next()` replaced with `next(iter(...))`
- `squeezenet1_1(pretrained=True)` replaced with `weights='DEFAULT'`
- dataset downloads use HTTPS with `--no-check-certificate` (the course host's
  certificate chain is incomplete for command-line clients)
