import numpy as np
import pytest
import torch

from src.metrics import iou_numpy, iou_pytorch

d = 3


@pytest.fixture
def zeros_2d_array():
    return np.zeros((1, d, d))


@pytest.fixture
def zeros_2d_tensor():
    return torch.zeros((1, d, d))


@pytest.fixture
def ones_2d_array():
    return np.ones((1, d, d))


@pytest.fixture
def ones_2d_tensor():
    return torch.ones((1, d, d))


@pytest.fixture
def array_upper_left():
    return np.array([[[1, 1, 0], [1, 1, 0], [1, 1, 0]]])


@pytest.fixture
def tensor_upper_left():
    return torch.tensor([[[1, 1, 0], [1, 1, 0], [1, 1, 0]]])


@pytest.fixture
def array_mid():
    arr = np.zeros((1, d, d))
    arr[0, 1, 1] = 1
    return arr


@pytest.fixture
def tensor_mid():
    tensor = torch.zeros((1, d, d))
    tensor[0, 1, 1] = 1
    return tensor


@pytest.fixture
def array_3d_mid():
    arr = np.zeros((1, d, d, d))
    arr[0, 1, 1, 1] = 1
    return arr


@pytest.fixture
def tensor_3d_mid():
    tensor = torch.zeros((1, d, d, d))
    tensor[0, 1, 1, 1] = 1
    return tensor


@pytest.fixture
def zeros_3d_array():
    return np.zeros((1, d, d, d))


@pytest.fixture
def zeros_3d_tensor():
    return torch.zeros((1, d, d, d))


@pytest.fixture
def ones_3d_array():
    return np.ones((1, d, d, d))


@pytest.fixture
def ones_3d_tensor():
    return torch.ones((1, d, d, d))


def test_array_iou_equal(zeros_2d_array, ones_2d_array, array_mid, zeros_3d_array, array_3d_mid):
    assert iou_numpy(zeros_2d_array, zeros_2d_array) == 1
    assert iou_numpy(zeros_3d_array, zeros_3d_array) == 1
    assert iou_numpy(ones_2d_array, ones_2d_array) == 1
    assert iou_numpy(array_3d_mid, array_3d_mid) == 1
    assert iou_numpy(array_mid, array_mid) == 1


def test_tensor_iou_equal(zeros_2d_tensor, ones_2d_tensor, tensor_mid, zeros_3d_tensor, tensor_3d_mid):
    assert iou_pytorch(zeros_2d_tensor, zeros_2d_tensor) == 1
    assert iou_pytorch(zeros_3d_tensor, zeros_3d_tensor) == 1
    assert iou_pytorch(ones_2d_tensor, ones_2d_tensor) == 1
    assert iou_pytorch(tensor_3d_mid, tensor_3d_mid) == 1
    assert iou_pytorch(tensor_mid, tensor_mid) == 1


def test_array_iou_not_equal(zeros_2d_array, ones_2d_array, array_upper_left, array_mid, zeros_3d_array,
                             ones_3d_array, array_3d_mid):
    assert np.isclose(
        iou_numpy(array_mid, array_upper_left), 1 / 6
    )
    assert np.isclose(
        iou_numpy(zeros_2d_array, ones_2d_array), 0, atol=1e-5
    )
    assert np.isclose(
        iou_numpy(zeros_3d_array, ones_3d_array), 0, atol=1e-5
    )
    assert np.isclose(
        iou_numpy(array_3d_mid, ones_3d_array), 1 / 27,
    )


def test_tensor_iou_not_equal(zeros_2d_tensor, ones_2d_tensor, tensor_upper_left, tensor_mid, zeros_3d_tensor,
                              ones_3d_tensor, tensor_3d_mid):
    assert torch.isclose(
        iou_pytorch(tensor_mid, tensor_upper_left), torch.tensor(1 / 6),
    )
    assert torch.isclose(
        iou_pytorch(zeros_2d_tensor, ones_2d_tensor), torch.tensor(0.)
    )
    assert torch.isclose(
        iou_pytorch(zeros_3d_tensor, ones_3d_tensor), torch.tensor(0.)
    )
    assert torch.isclose(
        iou_pytorch(tensor_3d_mid, ones_3d_tensor), torch.tensor(1 / 27),
    )
