"""Data-free regression tests for EECS 498/598 Winter 2022 Assignment 2."""

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import torch


A2_ROOT = (
    Path(__file__).resolve().parents[1]
    / "assignments"
    / "a2-linear-classifiers"
)
sys.path.insert(0, str(A2_ROOT))

import linear_classifier as linear  # noqa: E402
import two_layer_net as network  # noqa: E402


LOOPS = (ast.For, ast.AsyncFor, ast.While)


def _has_loop(function):
    source = textwrap.dedent(inspect.getsource(function))
    return any(isinstance(node, LOOPS) for node in ast.walk(ast.parse(source)))


def _numeric_gradient(function, tensor, step=1e-6):
    gradient = torch.zeros_like(tensor)
    flat_tensor = tensor.view(-1)
    flat_gradient = gradient.view(-1)
    for index in range(flat_tensor.numel()):
        old_value = flat_tensor[index].item()
        flat_tensor[index] = old_value + step
        positive = function().item()
        flat_tensor[index] = old_value - step
        negative = function().item()
        flat_tensor[index] = old_value
        flat_gradient[index] = (positive - negative) / (2 * step)
    return gradient


def test_svm_naive_and_vectorized_match():
    torch.manual_seed(2)
    X = torch.randn(8, 6, dtype=torch.float64)
    y = torch.tensor([0, 1, 3, 2, 1, 0, 3, 2])
    W = 0.01 * torch.randn(6, 4, dtype=torch.float64)

    naive_loss, naive_gradient = linear.svm_loss_naive(W, X, y, reg=0.2)
    vector_loss, vector_gradient = linear.svm_loss_vectorized(W, X, y, reg=0.2)

    assert torch.allclose(naive_loss, vector_loss, atol=1e-12)
    assert torch.allclose(naive_gradient, vector_gradient, atol=1e-12)
    assert not _has_loop(linear.svm_loss_vectorized)


def test_softmax_naive_and_vectorized_match_and_are_stable():
    torch.manual_seed(3)
    X = 50 * torch.randn(9, 5, dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 1])
    W = 4 * torch.randn(5, 4, dtype=torch.float64)

    naive_loss, naive_gradient = linear.softmax_loss_naive(W, X, y, reg=0.1)
    vector_loss, vector_gradient = linear.softmax_loss_vectorized(
        W, X, y, reg=0.1
    )

    assert torch.isfinite(naive_loss) and torch.isfinite(vector_loss)
    assert torch.allclose(naive_loss, vector_loss, atol=1e-11)
    assert torch.allclose(naive_gradient, vector_gradient, atol=1e-11)
    assert not _has_loop(linear.softmax_loss_vectorized)


def test_linear_minibatch_training_and_prediction():
    torch.manual_seed(4)
    X = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    y = torch.tensor([0, 0, 1, 1])

    X_batch, y_batch = linear.sample_batch(X, y, len(X), batch_size=11)
    assert X_batch.shape == (11, 2)
    assert y_batch.shape == (11,)

    W, history = linear.train_linear_classifier(
        linear.softmax_loss_vectorized,
        None,
        X,
        y,
        learning_rate=0.2,
        reg=0.0,
        num_iters=200,
        batch_size=4,
    )
    assert history[-1] < history[0]
    assert torch.equal(linear.predict_linear_classifier(W, X), y)


def test_two_layer_forward_loss_and_gradients():
    torch.manual_seed(5)
    X = torch.randn(4, 3, dtype=torch.float64)
    y = torch.tensor([0, 2, 1, 2])
    params = {
        "W1": 0.2 * torch.randn(3, 5, dtype=torch.float64),
        "b1": 0.1 + 0.1 * torch.randn(5, dtype=torch.float64),
        "W2": 0.2 * torch.randn(5, 3, dtype=torch.float64),
        "b2": 0.1 * torch.randn(3, dtype=torch.float64),
    }

    scores, hidden = network.nn_forward_pass(params, X)
    expected_hidden = (X.mm(params["W1"]) + params["b1"]).clamp(min=0)
    assert torch.allclose(hidden, expected_hidden)
    assert torch.allclose(scores, hidden.mm(params["W2"]) + params["b2"])

    loss, gradients = network.nn_forward_backward(params, X, y, reg=0.05)
    assert loss.ndim == 0
    for name, analytic in gradients.items():
        numeric = _numeric_gradient(
            lambda: network.nn_forward_backward(params, X, y, reg=0.05)[0],
            params[name],
        )
        assert torch.allclose(analytic, numeric, atol=2e-7, rtol=2e-5)


def test_two_layer_training_and_prediction():
    torch.manual_seed(6)
    X = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    y = torch.tensor([0, 0, 1, 1])
    net = network.TwoLayerNet(2, 8, 2, dtype=torch.float64, device="cpu", std=0.1)
    stats = net.train(
        X,
        y,
        X,
        y,
        learning_rate=0.2,
        learning_rate_decay=1.0,
        reg=0.0,
        num_iters=300,
        batch_size=4,
    )
    assert stats["loss_history"][-1] < stats["loss_history"][0]
    assert torch.equal(net.predict(X), y)


def test_search_parameter_contracts():
    svm_lrs, svm_regs = linear.svm_get_search_params()
    softmax_lrs, softmax_regs = linear.softmax_get_search_params()
    nn_lrs, hidden_sizes, nn_regs, decays = network.nn_get_search_params()

    assert 5 <= len(svm_lrs) * len(svm_regs) <= 25
    assert 5 <= len(softmax_lrs) * len(softmax_regs) <= 25
    assert all(len(values) >= 2 for values in (nn_lrs, hidden_sizes, nn_regs, decays))
    assert len(nn_lrs) * len(hidden_sizes) * len(nn_regs) * len(decays) < 256
