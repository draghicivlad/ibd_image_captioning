import torch
import yaml
from PIL import Image

from sources.model import Baseline
from sources.utils import create_vocab_flickr30k, create_vocab_cocoro


config_lstm = {
    "data_root_path_en": "../data/flickr30k",
    "data_root_path_ro": "../data/coco/output",
    "language": "ro",
    "model_saved_path": "../outputs/lstm_dec_2lay11_16_30/ckpts/model-epoch=03.ckpt",
    "model_name": "ro_lstm_256_512_2_0.3",
    "encoder": {
        "name": "resnet18",
        "latent_dim": 256
    },
    "decoder": {
        "name": "lstm",
        "embed_size": 256,
        "hidden_size": 512,
        "num_layers": 2,
        "dropout_prob": 0.3
    },
    "lr": 0.001,
    "use_data_augmentation": False,
    "epochs": 15,
    "batch_size": 32
}

config_transf = {
    "data_root_path_en": "../data/flickr30k",
    "data_root_path_ro": "../data/coco/output",
    "language": "ro",
    "model_saved_path": "../outputs/lstm_dec_2lay11_16_30/ckpts/model-epoch=03.ckpt",
    "model_name": "ro_lstm_256_512_2_0.3",
    "encoder": {
        "name": "resnet18",
        "latent_dim": 256
    },
    "decoder": {
        "name": "transformer",
        "d_model": 256,
        "d_ff": 512,
        "nheads": 4,
        "num_layers": 2,
        "dropout_prob": 0.3
    },
    "lr": 0.001,
    "use_data_augmentation": False,
    "epochs": 15,
    "batch_size": 32
}

def load_model(path, vocab, config):
    model = Baseline(config, vocab)
    model.load_state_dict(torch.load(path)["state_dict"])

    model.eval()
    model.cuda()
    return model


class ModelInfer():
    def __init__(self):
        with open('config.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        en_vocab = create_vocab_flickr30k(config["data_root_path_en"])
        ro_vocab = create_vocab_cocoro(config["data_root_path_ro"])

        self.romanian_models_paths = {
            "transf_256_512_2_0.3" : [
                r"C:\Users\Vlad\ibd_image_captioning\outputs\ro_transf_256_512_2_0.3\ckpts\model-epoch=12.ckpt",
                ro_vocab, config_transf]
        }

        self.romanian_models = {}

        for k, v in self.romanian_models_paths.items():
            self.romanian_models[k] = (load_model(v[0], v[1], v[2]), v[2])

        self.english_models_paths = {
            "lstm_256_512_2_0.3": [
                r"C:\Users\Vlad\ibd_image_captioning\outputs\en_lstm_256_512_2_0.3\ckpts\model-epoch=03.ckpt",
                en_vocab, config_lstm]
        }

        self.english_models = {}

        for k, v in self.english_models_paths.items():
            self.english_models[k] = (load_model(v[0], v[1], v[2]), v[2])

    def caption_image_romanian(self, image_path):
        image = Image.open(image_path)
        output_list = []

        for k, v in self.romanian_models.items():
            model, config = v

            caption = model.caption_image(image, config)
            output_list.append(f"{k}: {caption}\n")

        return output_list

    def caption_image_english(self, image_path):
        image = Image.open(image_path)
        output_list = []

        for k, v in self.english_models.items():
            model, config = v

            caption = model.caption_image(image, config)
            output_list.append(f"{k}: {caption}\n")

        return output_list
