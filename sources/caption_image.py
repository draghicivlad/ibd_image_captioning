import torch
import yaml
from PIL import Image

from model import Baseline
from utils import create_vocab_flickr30k, create_vocab_cocoro

IMAGE_PATH = r"C:\Users\Vlad\Downloads\test3.jpeg"


def caption_image(config):
    if config["language"] == "en":
        vocab = create_vocab_flickr30k(config["data_root_path_en"])
    elif config["language"] == "ro":
        vocab = create_vocab_cocoro(config["data_root_path_ro"])

    model = Baseline(config, vocab)
    model.load_state_dict(torch.load(config["model_saved_path"])["state_dict"])

    model.eval()
    model.cuda()

    image = Image.open(IMAGE_PATH)

    caption = model.caption_image(image, config)
    print(caption)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    caption_image(config)
