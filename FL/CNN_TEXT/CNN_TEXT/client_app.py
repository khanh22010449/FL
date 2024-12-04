"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from CNN_TEXT.task import (
    TextCNN,
    get_weights,
    load_data,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloadder, valloadder, local_epochs):
        self.model = model
        self.trainloadder = trainloadder
        self.valloadder = valloadder
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train_loss = train(
            self.model,
            self.trainloadder,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.model),
            len(self.trainloadder),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.valloadder, self.device)
        return loss, len(self.valloadder), {"accuracy": accuracy}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, len_vocab = load_data(
        partition_id, num_partitions, 100000, 512
    )

    model = TextCNN(
        vocab_size=len_vocab,
        embedding_dim=256,
        num_filters=128,
        kernel_size=2,
        max_length=512,
    )

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(model, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
