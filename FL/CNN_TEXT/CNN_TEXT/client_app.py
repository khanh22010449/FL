"""pytorch: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy

from CNN_TEXT.task import (
    TextCNN,
    get_weights,
    load_data,
    set_weights,
    train,
    test,
)
from flwr.common.logger import log
from logging import INFO


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, client_state: RecordSet, trainloadder, valloadder, local_epochs
    ):
        self.model = model
        self.client_state = client_state
        self.trainloadder = trainloadder
        self.valloadder = valloadder
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        train_loss = train(
            self.model,
            self.trainloadder,
            self.local_epochs,
            self.device,
        )
        self._save_layer_weights_to_state()
        return (
            get_weights(self.model),
            len(self.trainloadder),
            {"train_loss": train_loss},
        )

    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        state_dict_arrays = {}
        for k, v in self.model.fc.state_dict().items():
            state_dict_arrays[k] = array_from_numpy(v.cpu().numpy())

        # Add to recordset (replace if already exists)
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(
            state_dict_arrays
        )

    def _load_layer_weights_from_state(self):
        """Load last layer weights to state."""
        if self.local_layer_name not in self.client_state.parameters_records:
            return

        state_dict = {}
        param_records = self.client_state.parameters_records
        for k, v in param_records[self.local_layer_name].items():
            state_dict[k] = torch.from_numpy(v.numpy())

        # apply previously saved classification head by this client
        self.model.fc.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        self._load_layer_weights_from_state()
        loss, accuracy = test(self.model, self.valloadder, self.device)
        return loss, len(self.valloadder), {"accuracy": accuracy}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, len_vocab = load_data(
        partition_id, num_partitions, 68000, 512
    )
    log(INFO, f"len vocab {len_vocab}")

    model = TextCNN(
        vocab_size=len_vocab,
        embedding_dim=256,
        num_filters=128,
        kernel_size=2,
        max_length=512,
    )

    local_epochs = context.run_config["local-epochs"]
    client_state = context.state

    # Return Client instance
    return FlowerClient(
        model, client_state, trainloader, valloader, local_epochs
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
