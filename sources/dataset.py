import os.path

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data


def make_collater(vocab):
    pad_value = vocab.stoi("<PAD>")

    def collate_fn(data):
        if isinstance(data[0], dict):  # Processor case
            # print("auzi in colataral")
            pixel_values = torch.stack([item['pixel_values'] for item in data])
            input_ids = pad_sequence([item['input_ids'] for item in data], batch_first=True, padding_value=pad_value)
            attention_mask = pad_sequence([item['attention_mask'] for item in data], batch_first=True, padding_value=0)
            return pixel_values, input_ids, attention_mask

        images = []
        captions = []
        pad_masks = []
        attention_masks = []

        if len(data[0]) > 3:
            # print("Problem in colataral")
            max_len = max([x.shape[0] for (_, _, _, x) in data])

            for image, caption, attention_mask, pad_mask in data:
                images.append(image)
                if isinstance(caption, torch.Tensor):
                    captions.append(caption.clone().detach())
                else:
                    captions.append(torch.tensor(caption, dtype=torch.long))
                if isinstance(attention_mask, torch.Tensor):
                    attention_masks.append(attention_mask.clone().detach())
                else:
                    attention_masks.append(torch.tensor(attention_mask, dtype=torch.long))

                pad_masks.append(torch.cat((pad_mask, torch.zeros(max_len - pad_mask.shape[0])), dim=-1))

            attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            images = torch.stack(images)
            captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_value)
            pad_masks = torch.stack(pad_masks)
            # print("End problem colataral")
            return images, captions_padded, attention_masks_padded, pad_masks

        else:
            max_len = max([x.shape[0] for (_, _, x) in data])
            for image, caption, pad_mask in data:
                images.append(image)
                if isinstance(caption, torch.Tensor):
                    captions.append(caption.clone().detach())
                else:
                    captions.append(torch.tensor(caption, dtype=torch.long))
                pad_masks.append(torch.cat((pad_mask, torch.zeros(max_len - pad_mask.shape[0])), dim=-1))

            images = torch.stack(images)
            captions_padded = pad_sequence(captions, batch_first=True, padding_value=pad_value)
            pad_masks = torch.stack(pad_masks)

            return images, captions_padded, pad_masks

    return collate_fn


class Flickr30k(data.Dataset):
    def __init__(self, vocab, root_path="../data/flickr30k", transforms=None, split="train", processor=None):
        self.root_path = root_path
        self.split = split
        self.movie_dir = "flickr30k-images"
        self.vocab = vocab
        self.transforms = transforms

        self.processor = processor

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

        if self.processor:
            encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt",
                                      legacy=False)

            encoding = {k: v.squeeze() for k, v in encoding.items()}

            return encoding['pixel_values'], encoding['input_ids'], encoding['attention_mask'], torch.ones(
                len(encoding['input_ids']))

        return (
            self.transforms(image),
            [self.vocab.stoi("<SOS>")] + [self.vocab.stoi(idx) for idx in caption.split()] + [self.vocab.stoi("<EOS>")],
            torch.ones(len(caption.split()) + 2)
        )


class COCORo(data.Dataset):
    def __init__(self, vocab, root_path="../data/coco/output", transforms=None, processor=None):
        self.root_path = root_path
        self.movie_dir = "images"
        self.vocab = vocab
        self.transforms = transforms

        self.processor = processor

        self.captions = pd.read_csv(os.path.join(self.root_path, "captions.token"), header=None, sep='\t')
        # print(self.captions.head())
        self.captions.reset_index(inplace=True, drop=True)

    def __len__(self):
        return len(self.captions.index)

    def __getitem__(self, item):
        row = self.captions.iloc[item]

        movie_name = row[0][:-2]
        caption = row[1][:1000].strip()

        image = Image.open(os.path.join(self.root_path, self.movie_dir, movie_name))

        if self.processor:
            encoding = self.processor(images=image, text=caption, padding="max_length", return_tensors="pt",
                                      legacy=False)

            encoding = {k: v.squeeze() for k, v in encoding.items()}

            return encoding

        return (
            self.transforms(image),
            [self.vocab.stoi("<SOS>")] + [self.vocab.stoi(idx) for idx in caption.split()] + [self.vocab.stoi("<EOS>")],
            torch.ones(len(caption.split()) + 2)
        )
