import math

import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.text import Perplexity, BLEUScore, ROUGEScore
from torchvision.models import ResNet18_Weights, resnet18, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from transformers import AutoProcessor, GitForCausalLM


class LSTMDecoder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(self.vocab), config["embed_size"])
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"], config["num_layers"], batch_first=True)
        self.fc = nn.Linear(config["hidden_size"], len(self.vocab))
        self.dropout = nn.Dropout(config["dropout_prob"])

    def forward(self, features, captions, pad_mask):
        captions = captions[:, :-1].contiguous()

        embeddings = self.dropout(self.embed(captions))  # [batch_size, seq_len, embed_size]
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # [batch_size, seq_len+1, embed_size]
        lstm_out, _ = self.lstm(inputs)  # [batch_size, seq_len+1, hidden_size]
        outputs = self.fc(lstm_out)
        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDecoderHead(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(self.vocab), config["d_model"])
        self.pe = PositionalEncoding(config["d_model"])

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["nheads"],
            dim_feedforward=config["d_ff"],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["num_layers"])

        self.fc = nn.Linear(config["d_model"], len(self.vocab))
        self.dropout = nn.Dropout(config["dropout_prob"])

    def forward(self, features, captions, pad_mask):
        captions = captions[:, :-1].contiguous()
        pad_mask = pad_mask[:, :-1].contiguous()

        embeddings = self.dropout(self.embed(captions))  # [batch_size, seq_len, embed_size]
        seq_len = embeddings.shape[1]

        decoder_out = self.decoder(
            tgt=embeddings,
            memory=features.unsqueeze(1),
            tgt_mask=torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(embeddings.device),
            tgt_key_padding_mask=~pad_mask.bool(),
            tgt_is_causal=True
        )
        outputs = self.fc(decoder_out)

        return outputs


class Baseline(L.LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.perplexity = Perplexity(ignore_index=self.vocab.stoi('<PAD>'))
        self.bleu = BLEUScore(smooth=True, n_gram=1)
        self.rouge = ROUGEScore()

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
            self.decoder = TransformerDecoderHead(config["decoder"], self.vocab)
        else:
            raise ValueError("Expected: lstm, transformer; Got: " + config["decoder"]["name"])

    def forward(self, images, captions, pad_mask):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, pad_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        images, captions, pad_mask = batch
        targets_input = captions

        if isinstance(self.decoder, TransformerDecoderHead):
            targets_output = captions[:, 1:].contiguous()
        else:
            targets_output = captions

        preds = self.forward(images=images, captions=targets_input, pad_mask=pad_mask)

        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, len(self.vocab)),
            targets_output.view(-1),
            ignore_index=self.vocab.stoi("<PAD>")
        )
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """
        This method defines what happens in a single batch during testing.
        """
        images, captions, pad_mask = batch
        targets_input = captions
        targets_output = captions

        # Forward pass
        preds = self.forward(images=images, captions=targets_input, pad_mask=pad_mask)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(
            preds.view(-1, len(self.vocab)),
            targets_output.view(-1),
            ignore_index=self.vocab.stoi("<PAD>")
        )

        # Compute perplexity (optional, if you want it logged during testing)
        self.perplexity.update(preds, targets_output)

        text_pred = torch.argmax(preds, dim=-1).cpu().tolist()
        text_target = targets_output.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos(idx) for idx in results_caption if idx not in (0, 1, 2, 3)])
            for results_caption in text_pred
        ]

        text_target = [
            " ".join([self.vocab.itos(idx) for idx in results_caption if idx not in (0, 1, 2, 3)])
            for results_caption in text_target
        ]

        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)

        # Log the test loss
        self.log("test_loss", loss, prog_bar=True)

        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """
        This method computes metrics after all test batches have been processed.
        """
        test_perplexity = self.perplexity.compute()
        self.log("test_perplexity", test_perplexity, prog_bar=True)
        self.perplexity.reset()
        self.log("test_bleu", self.bleu.compute())
        self.bleu.reset()
        self.log("test_rouge", self.rouge.compute()['rougeL_fmeasure'])
        self.rouge.reset()

    def validation_step(self, batch, batch_idx):
        images, captions, pad_mask = batch
        targets_input = captions

        if isinstance(self.decoder, TransformerDecoderHead):
            targets_output = captions[:, 1:].contiguous()
        else:
            targets_output = captions

        preds = self.forward(images=images, captions=targets_input, pad_mask=pad_mask)

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

        text_pred = torch.argmax(preds, dim=-1).cpu().tolist()
        text_target = targets_output.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos(idx) for idx in results_caption if idx not in (0, 1, 2, 3)])
            for results_caption in text_pred
        ]

        text_target = [
            " ".join([self.vocab.itos(idx) for idx in results_caption if idx not in (0, 1, 2, 3)])
            for results_caption in text_target
        ]

        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)

    def on_validation_epoch_end(self):
        self.log("val_perplexity", self.perplexity.compute())
        self.perplexity.reset()
        self.log("val_bleu", self.bleu.compute())
        self.bleu.reset()
        self.log("val_rouge", self.rouge.compute()['rougeL_fmeasure'])
        self.rouge.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config["epochs"], eta_min=1e-6)

        return [optimizer], [scheduler]

    def caption_image(self, image, config, max_length=50):
        if config["encoder"]["name"].lower() == "resnet18":
            transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
        elif config["encoder"]["name"].lower() == "resnet34":
            transforms = ResNet34_Weights.IMAGENET1K_V1.transforms()
        elif config["encoder"]["name"].lower() == "resnet50":
            transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
        else:
            raise ValueError("Expected: resnet18, resnet34, resnet50; Got: " + config["encoder"]["name"])

        image = transforms(image)
        image = image.unsqueeze(0).to(self.device)
        results_caption = [self.vocab.stoi("<SOS>")]

        with torch.no_grad():
            features = self.encoder(image)

            for _ in range(max_length):
                captions = torch.tensor(results_caption).unsqueeze(0).to(torch.int64).to(self.device)
                embeddings = self.decoder.dropout(self.decoder.embed(captions))  # [batch_size, seq_len, embed_size]

                if isinstance(self.decoder, TransformerDecoderHead):
                    decoder_out = self.decoder.decoder(
                        tgt=embeddings,
                        memory=features.unsqueeze(1),
                    )
                else:
                    inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # [batch_size, seq_len+1, embed_size]
                    decoder_out, _ = self.decoder.lstm(inputs)  # [batch_size, seq_len+1, hidden_size]

                preds = self.decoder.fc(decoder_out)
                preds = torch.argmax(preds[0, -1, :]).cpu().item()
                results_caption.append(preds)

                if self.vocab.itos(preds) == "<EOS>":
                    break

        text = " ".join([self.vocab.itos(idx) for idx in results_caption])
        text = text.replace("<SOS>", "")
        text = text.replace("<EOS>", "")
        text = text.strip()

        return text


class DictAsVocab:
    def __init__(self, vocab_dict):
        self.tokens = vocab_dict
        self.stoi = vocab_dict
        self.itos = {v: k for k, v in vocab_dict.items()}

    def __len__(self):
        return len(self.tokens)


class FineTuneTeacherModel(L.LightningModule):
    def __init__(self, vocab, config, model_name, processor):
        super().__init__()
        self.config = config
        # self.vocab = vocab
        self.model = GitForCausalLM.from_pretrained(model_name)

        for param in self.model.git.parameters():
            # print(param)
            param.requires_grad = False

        # for name, layer in self.model.named_children():
        #     print(name)

        # self.vocab_size = len(vocab)

        # Modify the output layer
        # self.model.output = torch.nn.Linear(self.model.output.in_features, self.vocab_size)

        # # Reinitialize the weights of the new output layer
        # torch.nn.init.xavier_uniform_(self.model.output.weight)

        # print(self.model.output)

        self.processor = processor
        self.vocab = DictAsVocab(self.processor.tokenizer.get_vocab())

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.processor.tokenizer.pad_token_id)

        self.perplexity = Perplexity()  # ignore_index=self.vocab.stoi['<PAD>'])
        self.bleu = BLEUScore(smooth=True, n_gram=1)
        self.rouge = ROUGEScore()

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.model(pixel_values=pixel_values, labels=input_ids)
        return outputs.logits

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            pixel_values, input_ids, attention_mask = batch
        else:
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                             labels=input_ids)
        loss = outputs.loss

        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            pixel_values, input_ids, attention_mask = batch
        else:
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                             labels=input_ids)
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True, logger=True)

        preds = outputs.logits

        seq_len = input_ids.size(1)
        preds = preds[:, :seq_len, :]  # Truncate predictions to match the ground truth sequence length

        self.perplexity.update(preds, input_ids)

        # Convert predictions and targets to text
        text_pred = torch.argmax(preds, dim=-1).cpu().tolist()
        text_target = input_ids.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos[idx] for idx in pred])
            # if 0 <= idx < len(self.vocab.tokens) if idx not in (0, 1, 2, 3)])
            for pred in text_pred
        ]
        text_target = [
            " ".join([self.vocab.itos[idx] for idx in target])
            # if 0 <= idx < len(self.vocab.tokens) if idx not in (0, 1, 2, 3)])
            for target in text_target
        ]

        # BLEU and ROUGE metrics
        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)

        return loss

    def on_validation_epoch_end(self):
        self.log("val_perplexity", self.perplexity.compute(), logger=True)
        self.perplexity.reset()
        self.log("val_bleu", self.bleu.compute(), logger=True)
        self.bleu.reset()
        self.log("val_rouge", self.rouge.compute()['rougeL_fmeasure'], logger=True)
        self.rouge.reset()

    def test_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            pixel_values, input_ids, attention_mask = batch
        else:
            pixel_values = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                             labels=input_ids)
        loss = outputs.loss

        self.log("test_loss", loss, prog_bar=True, logger=True)

        # Update perplexity
        preds = outputs.logits
        seq_len = input_ids.size(1)
        preds = preds[:, :seq_len, :]

        self.perplexity.update(preds, input_ids)

        # Convert predictions and targets to text for BLEU and ROUGE
        text_pred = torch.argmax(preds, dim=-1).cpu().tolist()
        text_target = input_ids.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos[idx] for idx in pred])
            # if 0 <= idx < len(self.vocab.tokens) if idx not in (0, 1, 2, 3)])
            for pred in text_pred
        ]
        text_target = [
            " ".join([self.vocab.itos[idx] for idx in target])
            # if 0 <= idx < len(self.vocab.tokens) if idx not in (0, 1, 2, 3)])
            for target in text_target
        ]

        # Update BLEU and ROUGE
        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)

        return loss

    def on_test_epoch_end(self):
        """
        This method computes metrics after all test batches have been processed.
        """
        test_perplexity = self.perplexity.compute()
        self.log("test_perplexity", test_perplexity, prog_bar=True, logger=True)
        self.perplexity.reset()
        self.log("test_bleu", self.bleu.compute(), logger=True)
        self.bleu.reset()
        self.log("test_rouge", self.rouge.compute()['rougeL_fmeasure'], logger=True)
        self.rouge.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

    def caption_image(self, image, config, max_length=50):
        with torch.no_grad():
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            device = next(self.model.parameters()).device  # Get the model's device
            pixel_values = pixel_values.to(device)

            generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_caption)

        return generated_caption


class KnowledgeDistillationModel(L.LightningModule):
    def __init__(self, teacher_model, config, vocab, pretrained_student_path=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.config = config
        # self.vocab = vocab
        # self.processor = processor
        self.vocab = DictAsVocab(self.teacher_model.processor.tokenizer.get_vocab())

        self.perplexity = Perplexity(ignore_index=0)  # self.vocab.stoi['<PAD>'])
        self.bleu = BLEUScore(smooth=True, n_gram=1)
        self.rouge = ROUGEScore()

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

        for param in self.teacher_model.model.git.parameters():
            param.requires_grad = False
        for param in self.teacher_model.model.output.parameters():
            param.requires_grad = False

        # DECODER
        if config["decoder"]["name"].lower() == "lstm":
            self.decoder = LSTMDecoder(config["decoder"], self.vocab)
        elif config["decoder"]["name"].lower() == "transformer":
            self.decoder = TransformerDecoderHead(config["decoder"], self.vocab)
        else:
            raise ValueError("Expected: lstm, transformer; Got: " + config["decoder"]["name"])

        # print("Aici", pretrained_student_path)
        # self.student = Baseline.load_from_checkpoint(config["pretrained_student_path"], config=config, vocab=vocab)
        if pretrained_student_path:
            self.student = Baseline.load_from_checkpoint(config["pretrained_student_path"], config=config, vocab=vocab)
            # self.encoder = self.student.encoder
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            # for param in self.encoder.fc.parameters():
            #     param.requires_grad = True
            self.decoder = self.student.decoder

            # for name, module in self.decoder.named_modules():
            #     print(f"Module Name: {name}")
            #     print(f"Module Type: {type(module)}\n")
            # # self.decoder.vocab = self.vocab
            self.decoder.vocab = self.vocab
            self.decoder.embed = nn.Embedding(len(self.vocab.tokens), config["decoder"]["d_model"])
            self.decoder.fc = nn.Linear(config["decoder"]["d_model"], len(self.vocab.tokens))

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # self.vocab.stoi["<PAD>"])
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, images, captions, pad_mask):
        # print("here in forward")
        features = self.encoder(images)
        # print("here in forward 2")
        outputs = self.decoder(features, captions, pad_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            images, input_ids, attention_mask, pad_mask = batch
        else:
            images = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pad_mask = batch["pad_mask"]
        # images, captions, pad_mask = batch
        targets_input = input_ids

        if isinstance(self.decoder, TransformerDecoderHead):
            targets_output = input_ids[:, 1:].contiguous()
        else:
            targets_output = input_ids

        # Forward pass for student
        student_preds = self.forward(images, targets_input, pad_mask)

        # Forward pass for teacher
        with torch.no_grad():
            teacher_preds = self.teacher_model.model(pixel_values=images, input_ids=targets_input,
                                                     attention_mask=attention_mask, labels=targets_input).logits

        teacher_preds = teacher_preds[:, :student_preds.shape[1], :]
        # Compute losses
        ce_loss = self.criterion(
            student_preds.view(-1, len(self.vocab)),
            targets_output.view(-1))
        kl_loss = self.kl_div(
            torch.log_softmax(student_preds, dim=-1),
            torch.softmax(teacher_preds, dim=-1)
        )
        loss = ce_loss + self.config["alpha"] * kl_loss

        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            images, input_ids, attention_mask, pad_mask = batch
        else:
            images = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pad_mask = batch["pad_mask"]
        # images, captions, pad_mask = batch
        targets_input = input_ids

        if isinstance(self.decoder, TransformerDecoderHead):
            targets_output = input_ids[:, 1:].contiguous()
        else:
            targets_output = input_ids

        student_preds = self.forward(images, targets_input, pad_mask)

        with torch.no_grad():
            teacher_preds = self.teacher_model.model(pixel_values=images, input_ids=targets_input,
                                                     attention_mask=attention_mask, labels=targets_input).logits

        teacher_preds = teacher_preds[:, :student_preds.shape[1], :]
        # Compute losses
        ce_loss = torch.nn.functional.cross_entropy(
            student_preds.view(-1, len(self.vocab)),
            targets_output.view(-1)
            # ,ignore_index=self.vocab.stoi["<PAD>"]
        )

        # ce_loss = self.criterion(student_preds.view(-1, len(self.vocab)), targets_output.view(-1))
        kl_loss = self.kl_div(
            torch.log_softmax(student_preds, dim=-1),
            torch.softmax(teacher_preds, dim=-1)
        )
        loss = ce_loss + self.config["alpha"] * kl_loss

        self.log("val_loss", loss, prog_bar=True, logger=True)

        self.perplexity.update(
            student_preds,
            targets_output
        )

        text_pred = torch.argmax(student_preds, dim=-1).cpu().tolist()
        text_target = targets_output.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos[idx] for idx in results_caption])  # if idx not in (0, 1, 2, 3)])
            for results_caption in text_pred
        ]

        text_target = [
            " ".join([self.vocab.itos[idx] for idx in results_caption])  # if idx not in (0, 1, 2, 3)])
            for results_caption in text_target
        ]

        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_perplexity", self.perplexity.compute())
        self.perplexity.reset()
        self.log("val_bleu", self.bleu.compute())
        self.bleu.reset()
        self.log("val_rouge", self.rouge.compute()['rougeL_fmeasure'])
        self.rouge.reset()

    def test_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            images, input_ids, attention_mask, pad_mask = batch
        else:
            images = batch["pixel_values"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pad_mask = batch["pad_mask"]
        # images, captions, pad_mask = batch
        targets_input = input_ids

        if isinstance(self.decoder, TransformerDecoderHead):
            targets_output = input_ids[:, 1:].contiguous()
        else:
            targets_output = input_ids

        student_preds = self.forward(images, targets_input, pad_mask)

        # with torch.no_grad():
        #     teacher_preds = self.teacher_model.model(pixel_values=images, input_ids=targets_input, attention_mask=attention_mask, labels=targets_input).logits

        # teacher_preds = teacher_preds[:, :student_preds.shape[1], :]
        # # Compute losses
        loss = self.criterion(student_preds.view(-1, len(self.vocab)), targets_output.view(-1))
        # kl_loss = self.kl_div(
        #     torch.log_softmax(student_preds, dim=-1),
        #     torch.softmax(teacher_preds, dim=-1)
        # )
        # loss = ce_loss + self.config["alpha"] * kl_loss

        self.log("test_loss", loss, prog_bar=True, logger=True)

        self.perplexity.update(
            student_preds,
            targets_output
        )

        text_pred = torch.argmax(student_preds, dim=-1).cpu().tolist()
        text_target = targets_output.cpu().tolist()

        text_pred = [
            " ".join([self.vocab.itos[idx] for idx in results_caption])  # if idx not in (0, 1, 2, 3)])
            for results_caption in text_pred
        ]

        text_target = [
            " ".join([self.vocab.itos[idx] for idx in results_caption])  # if idx not in (0, 1, 2, 3)])
            for results_caption in text_target
        ]

        self.bleu.update(text_pred, text_target)
        self.rouge.update(text_pred, text_target)
        return loss

    def on_test_epoch_end(self):
        self.log("test_perplexity", self.perplexity.compute())
        self.perplexity.reset()
        self.log("test_bleu", self.bleu.compute())
        self.bleu.reset()
        self.log("test_rouge", self.rouge.compute()['rougeL_fmeasure'])
        self.rouge.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config["epochs"], eta_min=1e-6)

        return [optimizer], [scheduler]

    def caption_image(self, image, config, max_length=50):
        if config["encoder"]["name"].lower() == "resnet18":
            transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
        elif config["encoder"]["name"].lower() == "resnet34":
            transforms = ResNet34_Weights.IMAGENET1K_V1.transforms()
        elif config["encoder"]["name"].lower() == "resnet50":
            transforms = ResNet50_Weights.IMAGENET1K_V1.transforms()
        else:
            raise ValueError("Expected: resnet18, resnet34, resnet50; Got: " + config["encoder"]["name"])

        image = transforms(image)
        image = image.unsqueeze(0).to(self.device)

        start_token = 'a'
        end_token = '.'

        results_caption = [self.vocab.stoi[start_token]]

        with torch.no_grad():
            features = self.encoder(image)

            for _ in range(max_length):
                captions = torch.tensor(results_caption).unsqueeze(0).to(torch.int64).to(self.device)
                embeddings = self.decoder.dropout(self.decoder.embed(captions))  # [batch_size, seq_len, embed_size]

                if isinstance(self.decoder, TransformerDecoderHead):
                    decoder_out = self.decoder.decoder(
                        tgt=embeddings,
                        memory=features.unsqueeze(1),
                    )
                else:
                    inputs = torch.cat((features.unsqueeze(1), embeddings),
                                       dim=1)  # [batch_size, seq_len+1, embed_size]
                    decoder_out, _ = self.decoder.lstm(inputs)  # [batch_size, seq_len+1, hidden_size]

                preds = self.decoder.fc(decoder_out)
                preds = torch.argmax(preds[0, -1, :]).cpu().item()
                results_caption.append(preds)

                if self.vocab.itos[preds] == end_token:
                    break

        return " ".join([self.vocab.itos[idx] for idx in results_caption])


if __name__ == "__main__":
    Baseline(config={"encoder": "resnet18"})
