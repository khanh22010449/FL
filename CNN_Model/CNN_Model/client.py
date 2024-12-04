"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from CNN_Model.task import (
    CNN,
    get_weights,
    load_data,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, local_epochs):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        train_loss = train(
            self.model,
            self.X_train,
            self.y_train,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.model),
            len(self.X_train),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.X_test, self.y_test, self.device)
        return loss, len(self.X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    model = CNN()

    X_train, y_train, X_test, y_test = load_data()
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(
        model, X_train, y_train, X_test, y_test, local_epochs
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
