import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dlSrc.models import get_model
from .dlSrc.schedulers import get_scheduler
from .dlSrc.optimizers import get_optimizer
from .dlSrc.criterions import get_criterion


class Segmentator(pl.LightningModule):

    def __init__(self, config):
        super(Segmentator, self).__init__()
        self.config = config

        self.model = get_model(config)
        self.criterion = get_criterion(config)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        # TODO write dataset
        self.dataset_train = None
        self.dataset_val = None

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
            batch_size=self.config.train.batch_size * 2,
            num_workers=self.config.train.num_workers,
        )

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output, target)
        # TODO add metric
        return {'val_loss': loss, 'val_IoU': 0}

    def validation_epoch_end(self, outputs: list):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}
