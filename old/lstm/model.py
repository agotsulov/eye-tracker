import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class TwoEyesWithLSTM(nn.Module):
    def __init__(self, num_classes=2, batch_size=50, seq_len=5):
        super(TwoEyesWithLSTM, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.2)
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.2)
        )
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2),
            nn.Dropout(0.2),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024 * 4 * 4 * 2 + 68 * 2, 1024),  # 1024 * 4 * 4 * 2 + 68 * 2
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2)
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lstm_layers = 4
        self.lstm = nn.LSTM(input_size=2048, hidden_size=1024, num_layers=self.lstm_layers, dropout=0.2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )


    def init_hidden(self, batch):
        return (torch.zeros(self.lstm_layers, self.batch_size, 1024).cuda(),
                torch.zeros(self.lstm_layers, self.batch_size, 1024).cuda())

    def forward(self, e_l, e_r, f):
        batch_size, seq_len, c, w, h = e_l.size()
        hidden = (torch.zeros(self.lstm_layers, seq_len, 1024).cuda(),
                  torch.zeros(self.lstm_layers, seq_len, 1024).cuda())

        left = e_l.view(batch_size * seq_len, c, w, h)
        right = e_r.view(batch_size * seq_len, c, w, h)
        left = self.conv_left(left)
        right = self.conv_right(right)
        left = left.view(batch_size, seq_len, -1)
        right = right.view(batch_size, seq_len, -1)
        out = torch.cat((left, right), 2)
        f = f.view(batch_size * seq_len, -1)
        f = self.lin1(f)
        f = f.view(batch_size, seq_len, -1)
        out = torch.cat((out, f), 2)
        out = self.lin2(out.view(batch_size * seq_len, -1))
        out = out.view(batch_size, seq_len, -1)
        out, hidden = self.lstm(out, hidden)
        # print(hidden[0].size())
        # print(out[:, -1, :].size())
        # out = torch.cat((out[], hidden[0], hidden[1]), 2)
        # print(hidden[0])
        out = self.fc(out[:, -1, :])

        return out

