import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast
from argparse import ArgumentParser


def get(kwargs, key, default=None):
    return kwargs[key] if key in kwargs else default


def mask_tokens(inputs, tokenizer, mask_prob=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling.
    :param inputs: Input tensor of shape (batch_size, seq_length, dimension)
    :param mask_token_id: ID for [MASK] token
    :param pad_token_id: ID for padding token
    :param mask_prob: Probability of masking each token
    :return: Tuple of masked inputs and labels
    """
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id

    labels = inputs.clone()

    # Create a mask to decide which tokens to mask
    probability_matrix = torch.full(labels.shape, mask_prob, device=inputs.device)

    # Create a mask to avoid padding tokens being masked
    special_tokens_mask = labels == pad_token_id
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Create a mask to avoid [CLS] and [SEP] tokens being masked
    special_tokens_mask = labels == mask_token_id
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Create mask tensor where each element is True if it should be masked, False otherwise
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set labels for masked tokens; set labels for non-masked tokens to -100
    labels[~masked_indices] = -100

    # Replace masked input tokens with the mask token id
    inputs[masked_indices] = mask_token_id

    return inputs, labels


class LSTMTypeInferenceModel(nn.Module):
    def __init__(
        self,
        tokenizer: BertTokenizerFast,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.parse_kwargs(kwargs)

        self.embedding = nn.Embedding(tokenizer.vocab_size, self.input_size)
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )
        self.fc = nn.Linear(
            self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            tokenizer.vocab_size,
        )

    def parse_kwargs(self, kwargs):
        self.input_size = get(kwargs, "input_size", 256)
        self.hidden_size = get(kwargs, "hidden_size", 512)
        self.num_layers = get(kwargs, "num_layers", 2)
        self.dropout = get(kwargs, "dropout", 0.5)
        self.bidirectional = get(kwargs, "bidirectional", True)

    def forward(self, masked_input_ids, lengths):
        x = self.embedding(masked_input_ids)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=masked_input_ids.shape[1])
        x = self.fc(x)
        return x

    def __repr__(self):
        return f"LSTMTypeInferenceModel(input_size={self.input_size}, hidden_size={self.hidden_size}, num_layers={self.num_layers}, dropout={self.dropout}, bidirectional={self.bidirectional})"


class LLVMIRDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
    ):
        self.dataset = self.create_data_list(dataset_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def collate_fn(self, batch):
        inputs = self.tokenizer(
            [item["llvm_ir"] for item in batch],
            max_length=4096,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def create_data_list(self, dataset_path):
        dataset_files = os.listdir(dataset_path)
        dataset = []
        for i, file in enumerate(dataset_files):
            print(f"Processing file {i+1}/{len(dataset_files)}", end="\r")

            with open(os.path.join(dataset_path, file), "r") as f:
                content = f.read()
                dataset.append({"llvm_ir": content})

        return dataset


class LLVMIRTypeInferenceModel(pl.LightningModule):
    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = tokenizer
        self.model = LSTMTypeInferenceModel(tokenizer, **kwargs)
        self.lr = get(kwargs, "lr", 1e-3)

    def forward(self, masked_input_ids, lengths):
        return self.model(masked_input_ids, lengths)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        lengths = attention_mask.cpu().sum(dim=1)
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids, lengths)
        loss = F.cross_entropy(
            output.view(-1, self.tokenizer.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        lengths = attention_mask.cpu().sum(dim=1)
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids, lengths)
        loss = F.cross_entropy(
            output.view(-1, self.tokenizer.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LLVMIRDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, tokenizer, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = LLVMIRDataset(self.dataset_path, self.tokenizer)
        self.collate_fn = dataset.collate_fn
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/small_dataset")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tokenizer_path", type=str, default="llvmir_tokenizer")
    parser.add_argument("--gpus", type=str, default="0,1")

    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()

    pl.seed_everything(2024)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    model = LLVMIRTypeInferenceModel(
        tokenizer=tokenizer,
        **args.__dict__,
    )

    dm = LLVMIRDataModule(args.dataset_path, tokenizer, args.batch_size)
    trainer = pl.Trainer(
        devices=args.gpus,
        max_epochs=args.max_epochs,
        logger=WandbLogger(
            project="llvmir_type_inference",
            name=model.model.__repr__(),
        ),
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"checkpoints/{model.model.__repr__()}",
            ),
            EarlyStopping(monitor="val_loss"),
        ],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
