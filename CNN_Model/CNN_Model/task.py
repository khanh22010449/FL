import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from collections import OrderedDict


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_features=3, output_size=1):
        super(CNN, self).__init__()
        # Biến đổi dữ liệu đầu vào thành dạng có thể xử lý bởi convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.fc1 = nn.Linear(32 * num_features, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, output_size)  # Output layer

    def forward(self, x):
        # Chuyển đổi tensor đầu vào thành dạng (batch_size, channels, num_features)
        x = x.unsqueeze(1)  # Thêm chiều channel (batch_size, 1, num_features)
        x = F.relu(self.conv1(x))  # Lớp tích chập 1
        x = F.relu(self.conv2(x))  # Lớp tích chập 2
        x = x.view(x.size(0), -1)  # Flatten tensor (batch_size, 32 * num_features)
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)  # Output layer
        return x.squeeze()


def load_data():
    X, y = make_regression(n_samples=100, n_features=3, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )
    train_X = torch.tensor(X_train, dtype=torch.float32)
    test_X = torch.tensor(X_test, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.float32)
    return train_X, test_X, train_y, test_y


def train(net, X_train, y_train, epochs: int, device):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    run_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(net(X_train.to(device)), y_train.to(device))
        loss.backward()
        optimizer.step()
        run_loss += loss.item()

    avg_trainloss = run_loss / len(X_train)
    # print(f"avg_trainloss  : {avg_trainloss}")
    return avg_trainloss


def test(net, X_test, y_test, device):
    net.to(device)
    criterion = nn.MSELoss().to(device)
    net.eval()
    with torch.no_grad():
        predictions = net(X_test.to(device))
        loss = criterion(predictions, y_test.to(device))

    # Chuyển đổi dữ liệu về CPU để tính toán các chỉ số khác
    predictions_np = predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Tính toán các chỉ số
    mae = mean_absolute_error(y_test_np, predictions_np)
    rmse = np.sqrt(loss.item())
    r2 = r2_score(y_test_np, predictions_np)

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
    loss = test(net, X_test, y_test, device)
