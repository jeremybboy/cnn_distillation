import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(64 * 7 * 7, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.pool(out)
        out = self.relu(out)
        f2 = self.conv2(out)
        out = self.relu(f2)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu(out)
        f4 = self.conv4(out)
        out = self.relu(f4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out, f2, f4