import torch
import torch.nn as nn


class TwoEyes(nn.Module):
    def __init__(self, num_classes=2, seq_len=5):
        super(TwoEyes, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3 * seq_len, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3 * seq_len, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
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
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        left = self.conv_left(e_l.view(batch_size, s * c, h, w))
        right = self.conv_right(e_r.view(batch_size, s * c, h, w))
        out = torch.cat((left, right), 1)
        f = f.view(batch_size, -1)
        f = self.lin1(f)
        out = out.view(batch_size, -1)
        out = torch.cat((out, f), 1)
        out = self.lin2(out)
        out = self.fc(out)
        return out


class EyeClassifier(nn.Module):
    def __init__(self, num_classes, seq_len):
        super(EyeClassifier, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3 * seq_len, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3 * seq_len, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
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
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        left = self.conv_left(e_l.view(batch_size, s * c, h, w))
        right = self.conv_right(e_r.view(batch_size, s * c, h, w))
        out = torch.cat((left, right), 1)
        f = f.view(batch_size, -1)
        f = self.lin1(f)
        out = out.view(batch_size, -1)
        out = torch.cat((out, f), 1)
        out = self.lin2(out)
        out = self.fc(out)
        return out


class EyeClassifierLSTM(nn.Module):
    def __init__(self, num_classes, seq_len):
        super(EyeClassifierLSTM, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2 * 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2 + 68 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )
        self.seq_len = seq_len
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=self.num_layers, batch_first=True)
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, 1024).cuda(),
                torch.zeros(self.num_layers, batch, 1024).cuda())

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        hidden = self.init_hidden(batch_size)
        left = self.conv_left(e_l.view(batch_size * s, c, h, w))
        right = self.conv_right(e_r.view(batch_size * s, c, h, w))
        out = torch.cat((left, right), 2)
        f = f.view(batch_size * s, -1)
        f = self.fc1(f)
        out = out.view(batch_size * s, -1)
        out = torch.cat((out, f), 1)
        out = self.fc2(out)
        out = out.view(batch_size, s, -1)
        out, hidden = self.lstm(out, hidden)
        out = self.fc3(out[:, -1, :])
        return out


class TwoEyesSameLayer(nn.Module):
    def __init__(self, num_classes=2, seq_len=5):
        super(TwoEyesSameLayer, self).__init__()
        self.conv_eyes = nn.Sequential(
            nn.Conv2d(3 * seq_len, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
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
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        left = self.conv_eyes(e_l.view(batch_size, s * c, h, w))
        right = self.conv_eyes(e_r.view(batch_size, s * c, h, w))
        out = torch.cat((left, right), 1)
        f = f.view(batch_size, -1)
        f = self.lin1(f)
        out = out.view(batch_size, -1)
        out = torch.cat((out, f), 1)
        out = self.lin2(out)
        out = self.fc(out)
        return out


class TwoEyesLSTM(nn.Module):
    def __init__(self, num_classes=2, seq_len=5):
        super(TwoEyesLSTM, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2 * 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2 + 68 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )
        self.seq_len = seq_len
        self.num_layers = 2
        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=self.num_layers, batch_first=True)
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, 1024).cuda(),
                torch.zeros(self.num_layers, batch, 1024).cuda())

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        hidden = self.init_hidden(batch_size)
        left = self.conv_left(e_l.view(batch_size * s, c, h, w))
        right = self.conv_right(e_r.view(batch_size * s, c, h, w))
        out = torch.cat((left, right), 2)
        f = f.view(batch_size * s, -1)
        f = self.fc1(f)
        out = out.view(batch_size * s, -1)
        out = torch.cat((out, f), 1)
        out = self.fc2(out)
        out = out.view(batch_size, s, -1)
        out, hidden = self.lstm(out, hidden)
        out = self.fc3(out[:, -1, :])
        return out


class TwoEyesRNN(nn.Module):
    def __init__(self, num_classes=2, seq_len=5):
        super(TwoEyesRNN, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2 * 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2 + 68 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )
        self.seq_len = seq_len
        self.num_layers = 2
        self.rnn = nn.RNN(input_size=2048, hidden_size=1024, num_layers=self.num_layers, batch_first=True)
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def init_hidden(self, batch):
        return torch.zeros(self.num_layers, batch, 1024).cuda()

    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        hidden = self.init_hidden(batch_size)
        left = self.conv_left(e_l.view(batch_size * s, c, h, w))
        right = self.conv_right(e_r.view(batch_size * s, c, h, w))
        out = torch.cat((left, right), 2)
        f = f.view(batch_size * s, -1)
        f = self.fc1(f)
        out = out.view(batch_size * s, -1)
        out = torch.cat((out, f), 1)
        out = self.fc2(out)
        out = out.view(batch_size, s, -1)
        out, hidden = self.rnn(out, hidden)
        out = self.fc3(out[:, -1, :])
        return out


class TwoEyesGRU(nn.Module):
    def __init__(self, num_classes=2, seq_len=5):
        super(TwoEyesGRU, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256)
        )
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2 * 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 2 + 68 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048)
        )
        self.seq_len = seq_len
        self.num_layers = 2
        self.gru = nn.GRU(input_size=2048, hidden_size=1024, num_layers=self.num_layers, batch_first=True)
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes),
        )

    def init_hidden(self, batch):
        return torch.zeros(self.num_layers, batch, 1024).cuda()


    def forward(self, e_l, e_r, f):
        batch_size, s, c, h, w = e_l.size()
        hidden = self.init_hidden(batch_size)
        left = self.conv_left(e_l.view(batch_size * s, c, h, w))
        right = self.conv_right(e_r.view(batch_size * s, c, h, w))
        out = torch.cat((left, right), 2)
        f = f.view(batch_size * s, -1)
        f = self.fc1(f)
        out = out.view(batch_size * s, -1)
        out = torch.cat((out, f), 1)
        out = self.fc2(out)
        out = out.view(batch_size, s, -1)
        out, hidden = self.gru(out, hidden)
        out = self.fc3(out[:, -1, :])
        return out


