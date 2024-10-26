"""pytorch: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAdagrad,FedAdam
from pytorch.task import Net, get_weights

arr = []

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net()) 
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy 
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0, 
        min_available_clients=2,
        initial_parameters=parameters,
    )
    # print(f"strategy : {strategy}")

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)





