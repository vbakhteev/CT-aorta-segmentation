import sys
import pytorch_lightning as pl

sys.path.append('.')

from src.segmentation.dl import Segmentator
from src.segmentation.dlSrc.config import load_config, parse_args


def main():
    args = parse_args()
    config = load_config(args.config_file)

    model = Segmentator(config)
    trainer = pl.Trainer(
        max_epochs=config.train.num_epochs,
        gpus=config.train.n_gpu,
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
