import numpy as np
import torch
import torch.nn as nn


class RBFOutput(nn.Module):

    def __init__(self, sigma: float = 1.0, digit_file: str = "DIGIT.npy"):
        super().__init__()
        proto = np.load(digit_file)
        centers = torch.tensor(proto.reshape(10, -1), dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.sigma = sigma

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        diff = h.unsqueeze(1) - self.centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=-1) 
        return torch.exp(-0.5 * dist2 / (self.sigma ** 2))


class LeNet5(nn.Module):

    def __init__(self, sigma: float = 1.0, digit_file: str = "DIGIT.npy"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)  

        self.fc1 = nn.Linear(120, 84)

        self.output = RBFOutput(sigma=sigma, digit_file=digit_file)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)

        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)

        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)  

        x = torch.tanh(self.fc1(x))
        x = self.output(x)
        return x


if __name__ == "__main__":
    model = LeNet5()
    dummy = torch.randn(8, 1, 32, 32)
    out = model(dummy)
    print("Output shape:", out.shape)
