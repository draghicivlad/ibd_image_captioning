import torch
import yaml
from PIL import Image

from model import Baseline
from utils import create_vocab_flickr30k

IMAGE_PATH = "../data/flickr30k/flickr30k-images/5519354264.jpg"

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    vocab = create_vocab_flickr30k(config["data_root_path"])

    model = Baseline(config, vocab)
    model.load_state_dict(torch.load(config["model_saved_path"])["state_dict"])

    model.eval()
    model.cuda()

    image = Image.open(IMAGE_PATH)

    caption = model.caption_image(image, config)
    print(caption)
