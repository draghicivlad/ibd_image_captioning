import os
import json
import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torchvision.transforms import v2

from dataset import Flickr30k, make_collater, COCORo
from model import Baseline, FineTuneTeacherModel, KnowledgeDistillationModel
from utils import create_vocab_flickr30k, create_vocab_cocoro
from transformers import AutoProcessor, AutoModelForCausalLM


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
            v2.Resize(size=256),
            v2.CenterCrop(size=(224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transforms


def load_datasets(config, processor=None):
    if config["language"] == "en":
        vocab = create_vocab_flickr30k(config["data_root_path_en"])
    elif config["language"] == "ro":
        vocab = create_vocab_cocoro(config["data_root_path_ro"])

    if config["language"] == "en":
        test_dataset = Flickr30k(vocab=vocab, root_path=config["data_root_path_en"],
                                 transforms=get_transform(config), split="test", processor=processor)
    elif config["language"] == "ro":
        dataset = COCORo(vocab=vocab, root_path=config["data_root_path_ro"],
                         transforms=get_transform(config, train=True))

        len_dataset = len(dataset)
        train_size = int(0.9 * len_dataset)
        val_size = len_dataset - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    collate_fn = make_collater(vocab)

    test_dl = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn,
                         num_workers=8)

    return test_dl, vocab


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

    logger, callbacks = init_logging()

    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)

    model_path = config["model_saved_path"]

    if config["teacher_finetuning"] or config["use_knowledge_distilation"]:
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", use_fast=True)
        test_dl, vocab = load_datasets(config, processor=processor)
    else:
        test_dl, vocab = load_datasets(config)

    if config["use_knowledge_distilation"]:
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", use_fast=True)
        teacher_model = FineTuneTeacherModel.load_from_checkpoint(config["pretrained_teacher_path"],
                                                                  config=config,
                                                                  vocab=vocab,
                                                                  model_name="microsoft/git-base-coco",
                                                                  processor=processor
                                                                  )
        model = KnowledgeDistillationModel.load_from_checkpoint(teacher_model, config, vocab,
                                                                pretrained_student_path=config[
                                                                    "pretrained_student_path"])
    elif config["teacher_finetuning"]:
        model = FineTuneTeacherModel.load_from_checkpoint(model_path,
                                                          config=config,
                                                          vocab=vocab,
                                                          model_name="microsoft/git-base-coco",
                                                          processor=processor
                                                          )
    else:
        model = Baseline.load_from_checkpoint(model_path, config=config, vocab=vocab)

    # Run testing
    test_results = trainer.test(model, dataloaders=test_dl)
    print("Test Results:", test_results)

    # Save results to a file
    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Test results saved to {output_file}")
