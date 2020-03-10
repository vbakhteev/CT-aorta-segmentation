from torch import optim

from .radam import RAdam


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    opt = optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)
    return opt


def adamw(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-2, amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    opt = optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay, amsgrad=amsgrad)
    return opt


def radam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0, degenerated_to_sgd=True, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    opt = RAdam(parameters, lr=lr, betas=betas, weight_decay=weight_decay, degenerated_to_sgd=degenerated_to_sgd)
    return opt


def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    opt = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    return opt


def get_optimizer(params, config):
    f = globals().get(config.optimizer.name)
    return f(params, **config.optimizer.params)
