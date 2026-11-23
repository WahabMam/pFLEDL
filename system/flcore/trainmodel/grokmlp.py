import torch.nn as nn


class GrokMLP(nn.Module):
    """Depth-3, width-200 ReLU MLP matching Omnigrok's (arXiv 2210.01117, Liu/Michaud/Tegmark)
    exact MNIST recipe: 784 -> 200 -> 200 -> 10, ReLU, raw linear output (no softmax/log-softmax --
    paired with MSE loss on one-hot targets, not CrossEntropy).
    """

    def __init__(self, in_features=784, hidden_dim=200, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x
