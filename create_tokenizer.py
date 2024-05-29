# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: BinAIDA-FngcVyLg
#     language: python
#     name: python3
# ---

from tokenizers import BertWordPieceTokenizer

# +
##### load dataset from dataset directory #####
import os
from datasets import Dataset

dataset = []
dataset_files = os.listdir("dataset/small_dataset")

for i, filename in enumerate(dataset_files):
    print(f'loading {filename} ({i+1}/{len(dataset_files)})', end="\r")

    ##### load only .ll files #####
    if filename.endswith(".ll"):
        with open("dataset/small_dataset/" + filename, "r") as f:
            data = f.read()
            dataset.append({"text": data})


dataset = Dataset.from_list(dataset)
# -

tokenizer = BertWordPieceTokenizer()
tokenizer.train_from_iterator(dataset["text"], vocab_size=30522)
os.makedirs("llvmir_tokenizer", exist_ok=True)
tokenizer.save_model("llvmir_tokenizer")
