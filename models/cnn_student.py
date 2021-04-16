import torch.nn as nn
import torch.nn.functional as F


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.linear = nn.Linear(64 * 7 * 7, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        fA = out
        out = self.relu(fA)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.pool(out)

        fB = out
        out = self.relu(fB)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out, fA, fB