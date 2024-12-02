import os

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from sources.dataset import Flickr30k, make_collater
from sources.model import Baseline
from sources.utils import create_vocab_flickr30k


def get_transform(config):
    if config["encoder"]["name"].lower() == "resnet18":
        return ResNet18_Weights.IMAGENET1K_V1.transforms()
    elif config["encoder"]["name"].lower() == "resnet34":
        return ResNet34_Weights.IMAGENET1K_V1.transforms()
    elif config["encoder"]["name"].lower() == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V1.transforms()
    else:
        raise ValueError("Expected: resnet18, resnet34, resnet50; Got: " + config["encoder"]["name"])


def load_datasets(config):
    vocab = create_vocab_flickr30k(config["data_root_path"])

    train_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path"],
                              transforms=get_transform(config), split="train")
    val_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path"],
                            transforms=get_transform(config), split="val")
    collate_fn = make_collater(vocab)

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, vocab


def init_model(config, vocab):
    model = Baseline(config, vocab)

    return model


def init_logging():

    # LOGGER
    logger_path = os.path.join("..", "outputs", "logs")
    logger = TensorBoardLogger(logger_path)

    # CALLBACKS
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join("..", "outputs", "ckpts"),
            filename='model-{epoch:02d}',
            mode="min",
        )
    )

    return logger, callbacks


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    train_dl, val_dl, vocab = load_datasets(config)

    model = init_model(config, vocab)
    logger, callbacks = init_logging()

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dl, val_dl)
