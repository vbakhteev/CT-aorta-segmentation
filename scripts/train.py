import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.append('.')

from src.segmentation.dl import Segmentator
from src.segmentation.dlSrc.config import load_config, parse_args


def main():
    args = parse_args()
    config = load_config(args.config_file)

    checkpoint_callback = ModelCheckpoint(
        filepath='weights/weights.ckpt',
        verbose=True, monitor='val_metric', mode='max',
    )

    model = Segmentator(config)
    trainer = pl.Trainer(
        max_epochs=config.train.num_epochs,
        gpus=config.train.n_gpu,
        checkpoint_callback=checkpoint_callback,
        weights_summary=None,
        progress_bar_refresh_rate=1,
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
