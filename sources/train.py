import datetime
import os

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.transforms import v2

from sources.dataset import Flickr30k, make_collater
from sources.model import Baseline
from sources.utils import create_vocab_flickr30k


def get_transform(config, train=False):
    if config["encoder"]["name"].lower() == "resnet18":
        transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
    elif config["encoder"]["name"].lower() == "resnet34":
        transforms = ResNet34_Weights.IMAGENET1K_V1.transforms()
    elif config["encoder"]["name"].lower() == "resnet50":
        transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
    else:
        raise ValueError("Expected: resnet18, resnet34, resnet50; Got: " + config["encoder"]["name"])

    if "use_data_augmentation" in config and config["use_data_augmentation"] and train:
        print("DATA AUGMENTATION IS USED!")
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms
        ])

    return transforms


def load_datasets(config):
    vocab = create_vocab_flickr30k(config["data_root_path"])

    train_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path"],
                              transforms=get_transform(config, train=True), split="train")
    val_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path"],
                            transforms=get_transform(config), split="val")
    collate_fn = make_collater(vocab)

    train_dl = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, vocab


def init_model(config, vocab):
    model = Baseline(config, vocab)

    return model


def init_logging(name):

    folder_name = name + datetime.datetime.now().strftime('%d_%H_%M')

    folder_path = os.path.join("..", "outputs", folder_name)
    logger_path = os.path.join("..", "outputs", folder_name, "logs")
    param_path = os.path.join("..", "outputs", folder_name, "ckpts")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    if not os.path.exists(param_path):
        os.makedirs(param_path)

    # LOGGER
    logger = TensorBoardLogger(logger_path)

    # CALLBACKS
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=param_path,
            filename='model-{epoch:02d}',
            mode="min",
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
        )
    ]

    return logger, callbacks


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    train_dl, val_dl, vocab = load_datasets(config)

    model = init_model(config, vocab)
    logger, callbacks = init_logging(config["model_name"])

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dl, val_dl)
