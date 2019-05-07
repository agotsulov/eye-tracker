import torch
import torch.nn as nn


class TwoEyes(nn.Module):
    def __init__(self, num_classes=2, batch_size=50, seq_len=5):
        super(TwoEyes, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3 * seq_len, 1024, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3 * seq_len, 1024, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2 * seq_len),
            nn.Linear(68 * 2 * seq_len, 68 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2 * 2)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2 + 68 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, e_l, e_r, f):
        batch_size, _, _, _ = e_l.size()
        left = self.conv_left(e_l)
        right = self.conv_right(e_r)
        out = torch.cat((left, right), 1)
        f = f.view(batch_size, -1)
        f = self.lin1(f)
        out = out.view(batch_size, -1)
        out = torch.cat((out, f), 1)
        out = self.lin2(out)
        out = self.fc(out)
        return out

