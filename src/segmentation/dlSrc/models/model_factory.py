import segmentation_models_pytorch as smp


def unet(**kwargs):
    model = smp.Unet(**kwargs)
    return model


def fpn(**kwargs):
    model = smp.FPN(**kwargs)
    return model


def linknet(**kwargs):
    model = smp.Linknet(**kwargs)
    return model


def pspnet(**kwargs):
    model = smp.PSPNet(**kwargs)
    return model


def get_model(config):
    if config.model.dims == 3:
        raise NotImplementedError(f'3-dim models are not implemented.')

    f = globals().get(config.model.name)
    return f(**config.model.params)
