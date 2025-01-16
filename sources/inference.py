import os
import json
import lightning as L
import yaml
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

from dataset import Flickr30k, make_collater, COCORo
from model import Baseline
from utils import create_vocab_flickr30k, create_vocab_cocoro


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
    if config["language"] == "en":
        vocab = create_vocab_flickr30k(config["data_root_path_en"])
    elif config["language"] == "ro":
        vocab = create_vocab_cocoro(config["data_root_path_ro"])

    if config["language"] == "en":
        test_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path_en"],
                                  transforms=get_transform(config), split="test")

    elif config["language"] == "ro":

        dataset = COCORo(vocab=vocab, root_path=config["data_root_path_ro"],
                         transforms=get_transform(config))
        batch_size = config["batch_size"]
        len_dataset = len(dataset)
        train_size = (int(0.7 * len_dataset) // batch_size) * batch_size
        test_size = (int(0.15 * len_dataset) // batch_size) * batch_size
        val_size = len_dataset - train_size - test_size

        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    collate_fn = make_collater(vocab)

    test_dl = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn,num_workers=8, drop_last=True)
    print(vocab)

    return  test_dl, vocab

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

    test_dl, vocab = load_datasets(config)

    # model = init_model(config, vocab)
    logger, callbacks = init_logging()

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)
    # trainer.fit(model, train_dl, val_dl)

    # Load the model from the checkpoint
    model_path = config["model_saved_path"]
    # model = Baseline.load_from_checkpoint(model_path, config=config, vocab=vocab)
    model = Baseline.load_from_checkpoint(model_path, config=config, vocab=vocab, strict=False)
    # Initialize the Lightning Trainer
    # trainer = Trainer()

    # Run testing
    test_results = trainer.test(model, dataloaders=test_dl)
    print("Test Results:", test_results)

    # Save results to a file
    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Test results saved to {output_file}")
