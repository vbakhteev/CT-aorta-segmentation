from pytorch_toolbelt import losses as L


def joint(first, second, first_weight=1.0, second_weight=1.0):
    return L.JointLoss(first, second, first_weight, second_weight)


def bce(**kwargs):
    return L.BCELoss(**kwargs)


def soft_bce(**kwargs):
    return L.SoftBCELoss(**kwargs)


def lovasz(**kwargs):
    return L.BinaryLovaszLoss(**kwargs)


def jaccard(**kwargs):
    return L.JaccardLoss(**kwargs)


def focal(**kwargs):
    return L.BinaryFocalLoss(**kwargs)


def dice(**kwargs):
    return L.DiceLoss(**kwargs)


def get_criterion(config):
    if config.criterion.name != 'joint':
        f = globals().get(config.criterion.name)
        return f(**config.criterion.params)

    else:
        f1 = globals().get(config.criterion.first)
        f2 = globals().get(config.criterion.second)
        criterion1 = f1(**config.criterion.first.params)
        criterion2 = f2(**config.criterion.second.params)
        return joint(
            criterion1, criterion2,
            config.criterion.params.w1,
            config.criterion.params.w2,
        )
