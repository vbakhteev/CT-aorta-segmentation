from torch.optim import lr_scheduler


def step(optimizer, last_epoch, step_size=80, gamma=0.1, **_):
    s = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
    return s


def multi_step(optimizer, last_epoch, milestones=(500, 5000), gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    s = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
    return s


def exponential(optimizer, last_epoch, gamma=0.995, **_):
    s = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    return s


def none(optimizer, last_epoch, **_):
    s = lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)
    return s


def reduce_lr_on_plateau(optimizer, last_epoch, mode='min', factor=0.1, patience=10,
                         threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=0, **_):
    s = lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience,
                                       threshold=threshold, threshold_mode=threshold_mode,
                                       cooldown=cooldown, min_lr=min_lr)
    return s


def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
    s = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    return s


def cosine_restarts(optimizer, last_epoch, T_0=5, T_mult=2, ):
    s = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
    return s


def get_scheduler(optimizer, config, last_epoch=-1):
    f = globals().get(config.scheduler.name)
    return f(optimizer, last_epoch, **config.scheduler.params)
