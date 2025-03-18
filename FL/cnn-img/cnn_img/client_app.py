"""CNN-IMG: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy
from cnn_img.task import Net, get_weights, load_data, set_weights, test, train

from flwr.common.logger import log
from logging import INFO


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net,client_state: RecordSet, trainloader, valloader, local_epochs):
        self.net = net
        self.client_state = client_state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        self._load_layer_weights_from_state()
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        self._save_layer_weights_to_state()
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )
    def _save_layer_weights_to_state(self):
        """Save last layer weights to state."""
        state_dict_arrays = {}
        for k, v in self.net.fc3.state_dict().items():
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
        self.net.fc3.load_state_dict(state_dict, strict=True)

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    client_state = context.state


    # Return Client instance
    return FlowerClient(net,client_state, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
