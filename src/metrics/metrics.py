import numpy as np
import torch

EPS = 1e-8


def iou_numpy(outputs: np.array, labels: np.array):
    """Intersection over union metric for numpy arrays
    size can be (BATCH, H, W) or size (BATCH, Z, H, W)
    Args:
        outputs (np.array): Outputs of model
        labels (np.array): Ground truth

    returns:
         np.array: vector of IoU for each sample
    """
    if outputs.shape != labels.shape:
        raise AttributeError(f'Shapes not equal: {outputs.shape} != {labels.shape}')
    outputs = outputs.astype(np.int8)
    labels = labels.astype(np.int8)

    bs = outputs.shape[0]
    outputs = outputs.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    intersection = (outputs & labels).sum(1)
    union = (outputs | labels).sum(1)

    iou = (intersection + EPS) / (union + EPS)
    return iou


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    """Intersection over union metric for torch tensors
        size can be (BATCH, H, W) or size (BATCH, Z, H, W)
        Args:
            outputs (torch.tensor): Outputs of model
            labels (torch.tensor): Ground truth

        returns:
             torch.tensor: vector of IoU for each sample
        """
    if outputs.shape != labels.shape:
        raise AttributeError(f'Shapes not equal: {outputs.shape} != {labels.shape}')
    outputs = outputs.int()
    labels = labels.int()

    bs = outputs.shape[0]
    outputs = outputs.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    intersection = (outputs & labels).sum(1).float()
    union = (outputs | labels).sum(1).float()

    iou = (intersection + EPS) / (union + EPS)
    return iou
