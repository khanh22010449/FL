import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_features=6, output_size=1):
        super(CNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Calculate the size after the convolutional layers
        # After two Conv1D layers with padding=1, the output size remains the same as the input size
        # num_features is the length of the input sequence
        self.flattened_size = (
            32 * num_features
        )  # This assumes no pooling layers are used

        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # Debugging line
        x = F.relu(self.conv1(x))  # First convolutional layer
        x = F.relu(self.conv2(x))  # Second convolutional layer
        x = x.view(x.size(0), -1)  # Flatten tensor (batch_size, 32 * num_features)
        print(f"Flattened shape: {x.shape}")  # Debugging line
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        print(f"Output shape: {x.shape}")  # Debugging line
        return x.squeeze()  # Remove any extra dimensions


# Example of how to create an instance of this model
# model = CNN(input_channels=1, num_features=6, output_size=1)


def load_data():
    # Load your data here (this is just an example)
    X, y = make_regression(n_samples=1000, n_features=6, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Reshape the input data to have shape (batch_size, input_channels, num_features)
    train_X = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (num_samples, 1, num_features)
    test_X = torch.tensor(X_test, dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (num_samples, 1, num_features)
    train_y = torch.tensor(
        y_train, dtype=torch.float32
    ).squeeze()  # Shape: (num_samples,)
    test_y = torch.tensor(
        y_test, dtype=torch.float32
    ).squeeze()  # Shape: (num_samples,)

    return train_X, train_y, test_X, test_y


def train(net, X_train, y_train, epochs: int, device):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    run_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = net(X_train.to(device))
        print(
            f"Model output shape: {outputs.shape}, Target shape: {y_train.shape}"
        )  # Debugging line
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

    avg_trainloss = run_loss / len(X_train)
    return avg_trainloss


def test(net, X_test, y_test, device):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    net.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Move data to the appropriate device
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # Forward pass
        predictions = net(X_test)
        print(
            f"Predictions shape: {predictions.shape}, Target shape: {y_test.shape}"
        )  # Debugging line

        # Calculate loss
        loss = criterion(predictions, y_test)

    return loss.item()


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    net = CNN()
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    avg_trainloss = train(net, X_train, y_train, epochs, device)
    print(avg_trainloss)
    loss = test(net, X_test, y_test, device)
