import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ..metrics import iou_pytorch
from .dlSrc.models import get_model
from .dlSrc.schedulers import get_scheduler
from .dlSrc.optimizers import get_optimizer
from .dlSrc.criterions import get_criterion
from .dlSrc.dataset import TrainDataset, ValidDataset, get_augmentations


class Segmentator(pl.LightningModule):

    def __init__(self, config=None, model=None):
        super(Segmentator, self).__init__()

        if config:
            self.config = config
            self.model = get_model(config)
            self.criterion = get_criterion(config)
        else:
            self.model = model

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        df = pd.read_csv(os.path.join(self.config.data.path, 'train.csv'))
        df_train = df[df['train']]
        df_train = df[~df['train']]
        augmentations = get_augmentations()
        self.dataset_train = TrainDataset(df_train, self.config, transform=augmentations)
        self.dataset_val = ValidDataset(df_train, self.config)

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
        )

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target.unsqueeze(1))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target.unsqueeze(1))

        output = torch.sigmoid(output).squeeze(1)
        output = (output > 0.5).float() * 1
        iou = iou_pytorch(output, target).mean()
        return {'val_loss': loss, 'val_metric': iou}

    def validation_epoch_end(self, outputs: list):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_metric = torch.stack([x['val_metric'] for x in outputs]).mean()
        return {'val_loss': val_loss, 'val_metric': val_metric}
