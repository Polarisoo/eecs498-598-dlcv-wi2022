"""Data-free regression tests for EECS 498/598 Winter 2022 Assignment 3."""

import sys
from pathlib import Path

import torch


A3_ROOT = (
    Path(__file__).resolve().parents[1]
    / "assignments"
    / "a3-neural-networks"
)
sys.path.insert(0, str(A3_ROOT))

from fully_connected_networks import (  # noqa: E402
    Dropout,
    FullyConnectedNet,
    Linear,
    ReLU,
    TwoLayerNet,
    adam,
    rmsprop,
    sgd_momentum,
)
from convolutional_networks import (  # noqa: E402
    BatchNorm,
    Conv,
    DeepConvNet,
    MaxPool,
    SpatialBatchNorm,
    ThreeLayerConvNet,
    kaiming_initializer,
)


def _relative_error(actual, expected):
    denominator = torch.maximum(
        torch.tensor(1e-12, dtype=actual.dtype, device=actual.device),
        actual.abs() + expected.abs(),
    )
    return ((actual - expected).abs() / denominator).max().item()


def _numeric_gradient(function, tensor, step=1e-6):
    gradient = torch.zeros_like(tensor)
    flat_tensor = tensor.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    for index in range(flat_tensor.numel()):
        old_value = flat_tensor[index].item()
        flat_tensor[index] = old_value + step
        positive = function().item()
        flat_tensor[index] = old_value - step
        negative = function().item()
        flat_tensor[index] = old_value
        flat_gradient[index] = (positive - negative) / (2 * step)
    return gradient


def test_linear_and_relu_gradients():
    torch.manual_seed(30)
    x = torch.randn(2, 3, 2, dtype=torch.float64)
    w = torch.randn(6, 4, dtype=torch.float64)
    b = torch.randn(4, dtype=torch.float64)
    dout = torch.randn(2, 4, dtype=torch.float64)

    _, cache = Linear.forward(x, w, b)
    dx, dw, db = Linear.backward(dout, cache)
    objective = lambda: (Linear.forward(x, w, b)[0] * dout).sum()
    assert _relative_error(dx, _numeric_gradient(objective, x)) < 1e-7
    assert _relative_error(dw, _numeric_gradient(objective, w)) < 1e-7
    assert _relative_error(db, _numeric_gradient(objective, b)) < 1e-7

    relu_x = torch.randn(3, 4, dtype=torch.float64)
    relu_dout = torch.randn_like(relu_x)
    _, relu_cache = ReLU.forward(relu_x)
    relu_dx = ReLU.backward(relu_dout, relu_cache)
    relu_objective = lambda: (
        ReLU.forward(relu_x)[0] * relu_dout
    ).sum()
    assert _relative_error(
        relu_dx, _numeric_gradient(relu_objective, relu_x)
    ) < 1e-7


def test_fully_connected_models_pass_gradient_check():
    torch.manual_seed(31)
    labels = torch.tensor([0, 2])
    models = [
        TwoLayerNet(
            input_dim=6,
            hidden_dim=5,
            num_classes=3,
            weight_scale=0.1,
            reg=0.2,
            dtype=torch.float64,
        ),
        FullyConnectedNet(
            [5, 4],
            input_dim=6,
            num_classes=3,
            dropout=0.25,
            seed=123,
            weight_scale=0.1,
            reg=0.2,
            dtype=torch.float64,
        ),
    ]

    for model in models:
        inputs = torch.randn(2, 6, dtype=torch.float64)
        _, gradients = model.loss(inputs, labels)
        for name, parameter in model.params.items():
            numeric = _numeric_gradient(
                lambda: model.loss(inputs, labels)[0], parameter
            )
            assert _relative_error(gradients[name], numeric) < 2e-5


def test_dropout_and_update_rules():
    inputs = torch.ones(20000, dtype=torch.float64)
    output, _ = Dropout.forward(
        inputs, {"mode": "train", "p": 0.25, "seed": 7}
    )
    assert abs(output.mean().item() - 1.0) < 0.03
    assert abs((output == 0).double().mean().item() - 0.25) < 0.03
    assert torch.equal(
        Dropout.forward(inputs, {"mode": "test", "p": 0.25})[0],
        inputs,
    )

    weights = torch.tensor([1.0, -2.0], dtype=torch.float64)
    gradient = torch.tensor([0.5, -0.25], dtype=torch.float64)
    momentum_weights, _ = sgd_momentum(
        weights,
        gradient,
        {
            "learning_rate": 0.1,
            "momentum": 0.9,
            "velocity": torch.zeros_like(weights),
        },
    )
    assert torch.allclose(
        momentum_weights,
        torch.tensor([0.95, -1.975], dtype=torch.float64),
    )
    assert torch.isfinite(rmsprop(weights, gradient)[0]).all()
    adam_weights, adam_config = adam(
        weights, gradient, {"learning_rate": 0.1}
    )
    assert adam_config["t"] == 1
    assert torch.allclose(
        adam_weights,
        torch.tensor([0.9, -1.9], dtype=torch.float64),
        atol=1e-7,
    )


def test_naive_convolution_gradients():
    torch.manual_seed(32)
    x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    w = torch.randn(2, 2, 3, 3, dtype=torch.float64)
    b = torch.randn(2, dtype=torch.float64)
    conv_param = {"stride": 1, "pad": 1}
    out, cache = Conv.forward(x, w, b, conv_param)
    dout = torch.randn_like(out)
    dx, dw, db = Conv.backward(dout, cache)
    objective = lambda: (
        Conv.forward(x, w, b, conv_param)[0] * dout
    ).sum()

    assert _relative_error(dx, _numeric_gradient(objective, x)) < 2e-6
    assert _relative_error(dw, _numeric_gradient(objective, w)) < 2e-6
    assert _relative_error(db, _numeric_gradient(objective, b)) < 2e-6


def test_naive_max_pool_gradient():
    torch.manual_seed(33)
    x = torch.randn(1, 2, 4, 4, dtype=torch.float64)
    pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
    out, cache = MaxPool.forward(x, pool_param)
    dout = torch.randn_like(out)
    dx = MaxPool.backward(dout, cache)
    objective = lambda: (
        MaxPool.forward(x, pool_param)[0] * dout
    ).sum()
    assert _relative_error(dx, _numeric_gradient(objective, x)) < 2e-6


def test_batchnorm_and_spatial_batchnorm_gradients():
    torch.manual_seed(34)
    x = torch.randn(5, 4, dtype=torch.float64)
    gamma = torch.randn(4, dtype=torch.float64)
    beta = torch.randn(4, dtype=torch.float64)
    dout = torch.randn_like(x)
    out, cache = BatchNorm.forward(x, gamma, beta, {"mode": "train"})
    dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
    alt_dx, alt_dgamma, alt_dbeta = BatchNorm.backward_alt(dout, cache)
    objective = lambda: (
        BatchNorm.forward(x, gamma, beta, {"mode": "train"})[0] * dout
    ).sum()

    assert _relative_error(dx, _numeric_gradient(objective, x)) < 2e-6
    assert _relative_error(
        dgamma, _numeric_gradient(objective, gamma)
    ) < 2e-6
    assert _relative_error(
        dbeta, _numeric_gradient(objective, beta)
    ) < 2e-6
    assert _relative_error(dx, alt_dx) < 1e-9
    assert _relative_error(dgamma, alt_dgamma) < 1e-9
    assert _relative_error(dbeta, alt_dbeta) < 1e-9

    spatial_x = torch.randn(2, 3, 3, 2, dtype=torch.float64)
    spatial_gamma = torch.randn(3, dtype=torch.float64)
    spatial_beta = torch.randn(3, dtype=torch.float64)
    spatial_dout = torch.randn_like(spatial_x)
    _, spatial_cache = SpatialBatchNorm.forward(
        spatial_x, spatial_gamma, spatial_beta, {"mode": "train"}
    )
    spatial_dx, _, _ = SpatialBatchNorm.backward(
        spatial_dout, spatial_cache
    )
    spatial_objective = lambda: (
        SpatialBatchNorm.forward(
            spatial_x, spatial_gamma, spatial_beta, {"mode": "train"}
        )[0]
        * spatial_dout
    ).sum()
    assert _relative_error(
        spatial_dx, _numeric_gradient(spatial_objective, spatial_x)
    ) < 3e-6


def test_convolutional_models_integrate_all_layers():
    torch.manual_seed(35)
    inputs = torch.randn(2, 2, 6, 6, dtype=torch.float64)
    labels = torch.tensor([0, 2])
    models = [
        ThreeLayerConvNet(
            input_dims=(2, 6, 6),
            num_filters=2,
            filter_size=3,
            hidden_dim=4,
            num_classes=3,
            weight_scale=1e-2,
            reg=0.1,
            dtype=torch.float64,
        ),
        DeepConvNet(
            input_dims=(2, 6, 6),
            num_filters=[2, 3],
            max_pools=[0],
            batchnorm=False,
            num_classes=3,
            weight_scale=1e-2,
            reg=0.1,
            dtype=torch.float64,
        ),
        DeepConvNet(
            input_dims=(2, 6, 6),
            num_filters=[2, 3],
            max_pools=[0],
            batchnorm=True,
            num_classes=3,
            weight_scale="kaiming",
            reg=0.1,
            dtype=torch.float64,
        ),
    ]

    for model in models:
        scores = model.loss(inputs)
        loss, gradients = model.loss(inputs, labels)
        assert scores.shape == (2, 3)
        assert torch.isfinite(loss)
        assert set(gradients) == set(model.params)
        assert all(torch.isfinite(value).all() for value in gradients.values())


def test_kaiming_initializer_shapes_and_variance():
    torch.manual_seed(36)
    linear = kaiming_initializer(100, 200, dtype=torch.float64)
    convolution = kaiming_initializer(3, 8, K=3, dtype=torch.float64)
    assert linear.shape == (100, 200)
    assert convolution.shape == (8, 3, 3, 3)
    assert abs(linear.var(unbiased=False).item() - 2 / 100) < 0.004
