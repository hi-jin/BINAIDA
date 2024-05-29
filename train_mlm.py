import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast
from argparse import ArgumentParser


class LSTMTypeInferenceLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(output)


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
        return self.tokenizer(
            self.dataset[idx]["llvm_ir"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.model = LSTMTypeInferenceLSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, text):
        return self.model(text)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids)
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids)
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), labels.view(-1))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # Replace masked input tokens with tokenizer.mask_token_id
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels


class LLVMIRDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, tokenizer, batch_size):
        super().__init__()
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = LLVMIRDataset(self.dataset_path, self.tokenizer)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class LLVMIRTypeInferenceModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.model = LSTMTypeInferenceLSTMModel(vocab_size, embedding_dim, hidden_dim, n_layers, dropout)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

    def forward(self, text):
        return self.model(text)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids)
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), labels.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids, labels = mask_tokens(input_ids, self.tokenizer)
        output = self(input_ids)
        loss = nn.CrossEntropyLoss()(output.view(-1, self.vocab_size), labels.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="dataset/small_dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--tokenizer_path", type=str, default="llvmir_tokenizer")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    pl.seed_everything(2024)

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    model = LLVMIRTypeInferenceModel(
        tokenizer.vocab_size,
        args.embedding_dim,
        args.hidden_dim,
        args.n_layers,
        args.dropout,
    )

    dm = LLVMIRDataModule(args.dataset_path, tokenizer, args.batch_size)
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=WandbLogger(),
        callbacks=[ModelCheckpoint(monitor="val_loss"), EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
