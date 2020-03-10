import pytest
import segmentation_models_pytorch as smp
import torch
from easydict import EasyDict

from src.segmentation.dlSrc import get_model, get_scheduler, get_optimizer, get_criterion


@pytest.fixture
def config():
    cfg = EasyDict({
        'model': {
            'name': 'unet',
            'dims': 2,
            'params': {
                'encoder_name': 'resnet18',
                'in_channels': 1,
                'classes': 1,
            }
        },

        'scheduler': {
            'name': 'reduce_lr_on_plateau',
            'params': {
                'factor': 0.1,
                'patience': 10,
            }
        },

        'optimizer': {
            'name': 'sgd',
            'params': {
                'lr': 1e-3
            }
        },

        'criterion': {
            'name': 'jaccard',
            'params': {
                'mode': 'binary'
            }
        },
    })

    return cfg


def test_model(config):
    model = get_model(config)
    shape = (1, 1, 64, 64)
    out = model(torch.ones(shape))

    assert isinstance(model, smp.Unet)
    assert out.shape == shape


def test_optimizer(config):
    model = get_model(config)
    optimizer = get_optimizer(model.parameters(), config)

    assert isinstance(optimizer, torch.optim.SGD)


def test_criterion(config):
    criterion = get_criterion(config)

    outputs = torch.zeros((2, 1, 5, 5))
    labels = torch.ones((2, 5, 5))
    loss = criterion(outputs, labels)

    assert isinstance(loss, torch.Tensor)


def test_scheduler(config):
    model = get_model(config)
    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
