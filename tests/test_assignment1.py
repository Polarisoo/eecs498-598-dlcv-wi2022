"""Small, data-free regression suite for EECS 498/598 Winter 2022 A1."""

import ast
import inspect
import sys
import textwrap
from pathlib import Path

import pytest
import torch


A1_ROOT = (
    Path(__file__).resolve().parents[1]
    / "assignments"
    / "a1-pytorch-knn"
)
sys.path.insert(0, str(A1_ROOT))

import knn  # noqa: E402
import pytorch101 as p101  # noqa: E402


LOOPS = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def tree(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function)))


def assert_no_loops(function):
    assert not any(isinstance(node, LOOPS) for node in ast.walk(tree(function)))


def squared_distances(x_train, x_test):
    train = x_train.reshape(x_train.shape[0], -1)
    test = x_test.reshape(x_test.shape[0], -1)
    return ((train[:, None] - test[None, :]) ** 2).sum(dim=2)


def test_tensor_construction_mutation_and_count():
    x = p101.create_sample_tensor()
    assert torch.equal(x, torch.tensor([[0.0, 10.0], [100.0, 0.0], [0.0, 0.0]]))
    result = p101.mutate_tensor(
        x, [(0, 0), (2, 1), (0, 0)], [1, 2, 3]
    )
    assert result is x
    assert x[0, 0] == 3 and x[2, 1] == 2
    assert p101.count_tensor_elements(torch.empty(2, 0, 5)) == 0
    assert p101.count_tensor_elements(torch.tensor(4)) == 1


def test_pi_and_multiples_of_ten():
    assert torch.allclose(p101.create_tensor_of_pi(4, 5), torch.full((4, 5), 3.14))
    cases = [
        (5, 25, [10.0, 20.0]),
        (5, 7, []),
        (-25, 25, [-20.0, -10.0, 0.0, 10.0, 20.0]),
        (-9, 9, [0.0]),
    ]
    for start, stop, expected in cases:
        actual = p101.multiples_of_ten(start, stop)
        assert actual.dtype == torch.float64
        assert actual.tolist() == expected


def test_slicing_and_slice_assignment():
    x = torch.arange(1, 29).reshape(4, 7)
    outputs = p101.slice_indexing_practice(x)
    expected = (x[-1], x[:, 2:3], x[:2, :3], x[::2, 1::2])
    for actual, wanted in zip(outputs, expected):
        assert torch.equal(actual, wanted)
        assert actual.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()

    target = torch.full((6, 9), -99, dtype=torch.int64)
    result = p101.slice_assignment_practice(target)
    wanted = torch.tensor(
        [[0, 1, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2],
         [3, 4, 3, 4, 5, 5], [3, 4, 3, 4, 5, 5]]
    )
    assert result is target
    assert torch.equal(target[:4, :6], wanted)
    assert torch.equal(target[4:], torch.full((2, 9), -99))


def test_integer_indexing_and_one_hot():
    x = torch.arange(1, 31).reshape(5, 6)
    assert torch.equal(p101.shuffle_cols(x), x[:, [0, 0, 2, 1]])
    assert torch.equal(p101.reverse_rows(x), x.flip(0))
    assert torch.equal(p101.take_one_elem_per_col(x), x[[1, 0, 3], [0, 1, 2]])

    labels = [3, 0, 3, 1, 5]
    expected = torch.zeros(5, 6)
    expected[torch.arange(5), torch.tensor(labels)] = 1
    assert torch.equal(p101.make_one_hot(labels), expected)
    assert_no_loops(p101.make_one_hot)


def test_sum_reshape_and_row_min():
    x = torch.tensor([[-1, 2, 0], [0, 5, -3], [8, -9, 0]])
    assert p101.sum_positive_entries(x) == 15
    assert type(p101.sum_positive_entries(x)) is int

    y = p101.reshape_practice(torch.arange(24))
    expected = torch.tensor([
        [0, 1, 2, 3, 12, 13, 14, 15],
        [4, 5, 6, 7, 16, 17, 18, 19],
        [8, 9, 10, 11, 20, 21, 22, 23],
    ])
    assert torch.equal(y, expected)

    rows = torch.tensor([[2, -1, 7, -1], [5, 4, 3, 2]])
    snapshot = rows.clone()
    assert torch.equal(
        p101.zero_row_min(rows),
        torch.tensor([[2, 0, 7, -1], [5, 4, 3, 0]]),
    )
    assert torch.equal(rows, snapshot)


def test_batched_matrix_multiply_and_normalize():
    torch.manual_seed(498)
    x = torch.randn(4, 3, 5, dtype=torch.float64)
    y = torch.randn(4, 5, 2, dtype=torch.float64)
    expected = torch.bmm(x, y)
    assert torch.allclose(p101.batched_matrix_multiply(x, y, True), expected)
    assert torch.allclose(p101.batched_matrix_multiply(x, y, False), expected)
    assert_no_loops(p101.batched_matrix_multiply_noloop)

    data = torch.tensor(
        [[0.0, 30.0, 600.0], [1.0, 10.0, 200.0],
         [-1.0, 20.0, 400.0], [2.0, 50.0, 100.0]],
        dtype=torch.float64,
    )
    centered = data - data.mean(dim=0)
    wanted = centered / torch.sqrt(
        (centered ** 2).sum(dim=0) / (data.shape[0] - 1)
    )
    assert torch.allclose(p101.normalize_columns(data), wanted)
    assert_no_loops(p101.normalize_columns)


def test_challenges():
    xs = [
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([-3.0, 5.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 9.0, 4.0], dtype=torch.float64),
    ]
    lengths = torch.tensor([1, 2, 4])
    assert torch.allclose(
        p101.challenge_mean_tensors(xs, lengths),
        torch.tensor([1.0, 1.0, 4.0], dtype=torch.float64),
    )
    values = torch.tensor([8, -2, 8, 5, -2, 9, 5, 5])
    uniques, indices = p101.challenge_get_uniques(values)
    assert torch.equal(values[indices], uniques)
    for value, index in zip(uniques.tolist(), indices.tolist()):
        assert index == values.tolist().index(value)
    assert_no_loops(p101.challenge_mean_tensors)
    assert_no_loops(p101.challenge_get_uniques)


@pytest.mark.parametrize(
    "function",
    [knn.compute_distances_two_loops,
     knn.compute_distances_one_loop,
     knn.compute_distances_no_loops],
)
@pytest.mark.parametrize("shape", [(4,), (2, 3), (2, 2, 2)])
def test_distance_functions(function, shape):
    torch.manual_seed(2022)
    train = torch.randn((5, *shape), dtype=torch.float64)
    test = torch.randn((3, *shape), dtype=torch.float64)
    actual = function(train, test)
    assert actual.shape == (5, 3)
    assert torch.allclose(actual, squared_distances(train, test), atol=1e-9)


def test_distance_functions_integer_dtype_and_loop_counts():
    train = torch.tensor([[0, 0], [3, 4], [-1, 2]])
    test = torch.tensor([[0, 4], [2, 2]])
    expected = torch.tensor([[16, 8], [9, 5], [5, 9]])
    for function in (
        knn.compute_distances_two_loops,
        knn.compute_distances_one_loop,
        knn.compute_distances_no_loops,
    ):
        assert torch.equal(function(train, test), expected)

    loop_counts = [
        sum(isinstance(node, LOOPS) for node in ast.walk(tree(function)))
        for function in (
            knn.compute_distances_two_loops,
            knn.compute_distances_one_loop,
            knn.compute_distances_no_loops,
        )
    ]
    assert loop_counts == [2, 1, 0]


def test_predict_labels_and_tie_breaking():
    dists = torch.tensor([
        [0.3, 0.4, 0.1], [0.1, 0.5, 0.5], [0.4, 0.1, 0.2],
        [0.2, 0.2, 0.4], [0.5, 0.3, 0.3],
    ])
    labels = torch.tensor([0, 1, 0, 1, 2])
    assert torch.equal(knn.predict_labels(dists, labels, 3), torch.tensor([1, 0, 0]))

    tie_dists = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    tie_labels = torch.tensor([3, 1, 3, 1])
    assert knn.predict_labels(tie_dists, tie_labels, 4).item() == 1


def test_classifier_cross_validation_and_best_k():
    train = torch.tensor([[0.0], [0.6], [2.1], [4.0], [7.3], [12.0]])
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    classifier = knn.KnnClassifier(train, labels)
    assert classifier.x_train is train and classifier.y_train is labels
    predictions = classifier.predict(torch.tensor([[0.1], [3.9], [10.0]]))
    assert torch.equal(predictions, torch.tensor([0, 1, 2]))

    result = knn.knn_cross_validate(
        train, labels, num_folds=3, k_choices=[1, 3]
    )
    assert set(result) == {1, 3}
    assert all(len(scores) == 3 for scores in result.values())
    assert knn.knn_get_best_k({7: [80, 60], 3: [70, 70], 5: [65, 75]}) == 3


def test_gpu_function_source_and_optional_execution():
    source = inspect.getsource(p101.mm_on_gpu)
    assert source.count(".cuda()") >= 2 and ".cpu()" in source
    if torch.cuda.is_available():
        x = torch.randn(5, 7)
        w = torch.randn(7, 3)
        assert torch.allclose(p101.mm_on_gpu(x, w), x @ w, atol=1e-4)
