# Assignment 5: RNNs, Image Captioning, and Transformers

Official assignment page: https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment5.html

This directory contains the official A5 starter layout plus completed Python implementations for the captioning and Transformer exercises. The notebooks are preserved for JupyterLab or Colab use.

## Implemented files

- `rnn_lstm_captioning.py`: vanilla RNN step/sequence forward and backward, word embeddings, temporal softmax loss, CaptioningRNN forward/sample, LSTM, dot-product attention, and AttentionLSTM.
- `transformers.py`: tokenization helpers, looped and vectorized scaled dot-product attention, SelfAttention, MultiHeadAttention, LayerNormalization, FeedForwardBlock, EncoderBlock, DecoderBlock, masks, positional encodings, and Transformer forward pass.
- `eecs598/utils.py`: OpenCV is imported lazily inside the attention visualizer so core tests run on Windows environments where `cv2` DLL loading fails.

## Local JupyterLab notes

Start JupyterLab from this directory and skip the Google Drive / Colab setup cells when working locally:

```powershell
cd assignments/a5-rnn-transformers
jupyter lab
```

The notebooks still contain the official training and submission cells. For a real course submission, run the required training cells with a GPU runtime and fill in the requested student metadata.

## Validation

From the repository root:

```powershell
python -m pytest tests/test_assignment5.py -q
python -m pytest -q
```

Current local validation: `4 passed` for A5 and `44 passed` for the full repository.

## Artifact caveat

The official A5 submission expects trained checkpoints such as `rnn_lstm_attention_submission.pt` and `transformer.pt`. These GPU training artifacts were not generated in this CPU-oriented Codex pass, and `.pt` files are intentionally excluded from Git.
