import lightning as L
import torch
from torch import nn
from torchmetrics.text import Perplexity
from torchvision.models import ResNet18_Weights, resnet18, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights


class LSTMDecoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(self.vocab), config["embed_size"])
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"], config["num_layers"], batch_first=True)
        self.fc = nn.Linear(config["hidden_size"], len(self.vocab))
        self.dropout = nn.Dropout(config["dropout_prob"])

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))  # [batch_size, seq_len, embed_size]
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # [batch_size, seq_len+1, embed_size]
        lstm_out, _ = self.lstm(inputs)  # [batch_size, seq_len+1, hidden_size]
        outputs = self.fc(lstm_out)
        return outputs


class Baseline(L.LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.perplexity = Perplexity(ignore_index=self.vocab.stoi('<PAD>'))

        # ENCODER
        if config["encoder"]["name"].lower() == "resnet18":
            self.encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Linear(512, config["encoder"]["latent_dim"])
        elif config["encoder"]["name"].lower() == "resnet34":
            self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Linear(512, config["encoder"]["latent_dim"])
        elif config["encoder"]["name"].lower() == "resnet50":
            self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Linear(2048, config["encoder"]["latent_dim"])
        else:
            raise ValueError("Expected: resnet18, resnet34, resnet50; Got: " + config["encoder"]["name"])

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.fc.parameters():
            param.requires_grad = True

        # DECODER
        if config["decoder"]["name"].lower() == "lstm":
            self.decoder = LSTMDecoder(config["decoder"], self.vocab)
        elif config["decoder"]["name"].lower() == "transformer":
            self.decoder = LSTMDecoder(config["decoder"], self.vocab)
        else:
            raise ValueError("Expected: lstm, transformer; Got: " + config["decoder"]["name"])

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def training_step(self, batch, batch_idx):
        images, captions = batch
        targets_input = captions[:, :-1].contiguous()
        targets_output = captions

        preds = self.forward(images=images, captions=targets_input)

        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, len(self.vocab)),
            targets_output.view(-1),
            ignore_index=self.vocab.stoi("<PAD>")
        )
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        targets_input = captions[:, :-1].contiguous()
        targets_output = captions

        preds = self.forward(images=images, captions=targets_input)

        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, len(self.vocab)),
            targets_output.view(-1),
            ignore_index=self.vocab.stoi("<PAD>")
        )
        self.log("val_loss", loss)

        self.perplexity.update(
            preds,
            targets_output
        )

    def on_validation_epoch_end(self):
        self.log("val_perplexity", self.perplexity.compute())
        self.perplexity.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return [optimizer], []

    def caption_image(self, image, max_length=50):
        results_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                pred_lstm, states = self.decoder.lstm(x, states)
                output = self.decoder.fc(pred_lstm.unsqueeze(0))
                predicted = output.argmax(-1)
                results_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if self.vocab.itos(predicted.item()) == "<EOS>":
                    break

        return " ".join([self.vocab.itos(idx) for idx in results_caption])


if __name__ == "__main__":
    Baseline(config={"encoder": "resnet18"})
