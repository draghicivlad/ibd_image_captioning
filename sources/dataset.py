import os.path

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


def make_collater(vocab):
    pad_value = vocab.stoi("<PAD>")

    def collate_fn(data):
        images = []
        captions = []

        for image, caption in data:
            images.append(image)
            captions.append(torch.tensor(caption, dtype=torch.long))

        images = torch.stack(images)
        captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_value)

        return images, captions_padded

    return collate_fn


class Flickr30k(data.Dataset):
    def __init__(self, vocab, root_path="../data/flickr30k", transforms=None, split="train"):
        self.root_path = root_path
        self.split = split
        self.movie_dir = "flickr30k-images"
        self.vocab = vocab
        self.transforms = transforms

        assert split in ("train", "val", "test")

        with open(os.path.join(self.root_path, f"{split}.txt"), "r") as f:
            self.movie_ids = f.readlines()
        self.movie_ids = set([mvname.replace("\n", "") for mvname in self.movie_ids])
        self.captions = pd.read_csv(os.path.join(self.root_path, "results_20130124.token"), header=None, sep='\t')

        remove_indices = []
        for idx in self.captions.index:
            if (self.captions.iloc[idx])[0].split('.')[0] not in self.movie_ids:
                remove_indices.append(idx)

        self.captions.drop(remove_indices, inplace=True)
        self.captions.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.captions.index)

    def __getitem__(self, item):
        row = self.captions.iloc[item]

        combined_id = row[0]
        caption = row[1]

        movie_name = combined_id.split('#')[0]
        image = Image.open(os.path.join(self.root_path, self.movie_dir, movie_name))

        return self.transforms(image), [self.vocab.stoi("<SOS>")] + [self.vocab.stoi(idx) for idx in caption.split()] + [self.vocab.stoi("<EOS>")]
