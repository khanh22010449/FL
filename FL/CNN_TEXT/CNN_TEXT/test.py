import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import numpy as np
from collections import Counter
from datasets import load_dataset
from flwr.common.logger import log
from logging import INFO

import warnings
from collections import OrderedDict
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_data(vocab_size: int, max_length: int):

    data = load_dataset("stanfordnlp/imdb")["train"]

    # Build vocabulary based on the entire partition
    word_counts = Counter(word for text in data["text"] for word in text.split())
    vocab = {
        word: i + 1
        for i, (word, _) in enumerate(word_counts.most_common(vocab_size - 1))
    }
    vocab["<PAD>"] = 0  # Padding token

    while len(vocab) < 68000:
        vocab[f"<PAD_{len(vocab)}>"] = 0

    def tokenize_and_pad(batch):
        sequences = [
            [vocab.get(word, 0) for word in text.split()] for text in batch["text"]
        ]
        padded_sequences = [
            (
                seq + [0] * (max_length - len(seq))
                if len(seq) < max_length
                else seq[:max_length]
            )
            for seq in sequences
        ]

        batch["padded"] = np.array(padded_sequences, dtype=np.int64)
        return batch

    # Apply tokenization and padding
    data = data.map(tokenize_and_pad, batched=True)
    data = data.remove_columns("text")

    # Convert 'padded' and 'label' to tensors
    def apply_tensor(batch):
        batch["padded"] = torch.tensor(batch["padded"], dtype=torch.long)
        if "label" in batch:
            batch["label"] = torch.tensor(batch["label"], dtype=torch.float32)
        return batch

    # Apply tensor conversion
    train_test = data.train_test_split(test_size=0.2, seed=42)
    train_test = train_test.with_transform(apply_tensor)

    # Verify the type
    # print(f"train_test : {type(partition_train_test['train']['padded'])}")

    # Create DataLoader
    log(INFO, f"test.py")
    log(INFO, f"len vocab {len(vocab)}")

    valloader = DataLoader(train_test["test"], batch_size=32, num_workers=4)
    for batch in valloader:
        print(batch)
        break
    return valloader


if __name__ == "__main__":
    # for i in range(10):
    #     trainloader, valloader, len_vocab = load_data(i, 10, 100000, 512)
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {device}")
    #     net = TextCNN(
    #         vocab_size=len_vocab,
    #         embedding_dim=256,
    #         num_filters=128,
    #         kernel_size=2,
    #         max_length=512,
    #     )
    #     net.to(device)  # Move the model to the desired device
    #     train(net, trainloader, 10, device)
    #     validate(net, valloader, device)
    valloader = load_data(68000, 512)
