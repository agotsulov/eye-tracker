import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
from scipy import misc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 100
num_classes = 2
batch_size = 50
sequence_length = 10
learning_rate = 0.001


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dirname='./data'):
        self.eye_left = []
        self.eye_right = []
        self.face = []
        self.x = []
        self.y = []
        self.dirname = dirname
        self.size = 0

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        names = os.listdir(dirname)

        print(len(os.listdir(dirname)))
        print("LOADING DATA...")

        for index in range(len(os.listdir(dirname))):
            curr = names[index]
            frame = misc.imread(dirname + '/' + curr)

            rects = detector(frame, 0)

            eye_right = None
            eye_left = None
            shape = None

            if rects is None:
                if os.path.exists(dirname + '/' + curr):
                    os.remove(dirname + '/' + curr)
                print('NOT FOUND RECT: ' + curr)
                continue

            for (i, rect) in enumerate(rects):
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:42]]))
                eye_left = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

                (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[43:48]]))
                eye_right = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

            if eye_right is None and eye_left is None:
                if os.path.exists(dirname + '/' + curr):
                    os.remove(dirname + '/' + curr)
                print('NOT FOUND EYES: ' + curr)
                continue

            self.add(int(curr.split('_')[2]),
                     int(curr[:-4:].split('_')[4]),
                     cv2.resize(eye_left, (32, 32)).reshape((3, 32, 32)),
                     cv2.resize(eye_right, (32, 32)).reshape((3, 32, 32)),
                     shape)
        print("END LOAD DATA")

    def add(self, x, y, eye_left, eye_right, face):
        self.x.append(x)
        self.y.append(y)
        self.eye_left.append(eye_left)
        self.eye_right.append(eye_right)
        self.face.append(face)
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.from_numpy(self.eye_left[index]).float(), \
               torch.from_numpy(self.eye_right[index]).float(), \
               torch.from_numpy(self.face[index]).float(), \
               torch.from_numpy(np.array([self.x[index], self.y[index]])).float()


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
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
            nn.BatchNorm2d(1024)
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
            nn.BatchNorm2d(1024)
        )
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(68 * 2),
            nn.Linear(68 * 2, 68 * 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(68 * 2),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024 * 4 * 4 * 2 + 68 * 2, 2048), # 1024 * 4 * 4 * 2 + 68 * 2
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )
        # self.lstm = nn.LSTM(input_size=1024, hidden_size=1024)
        self.fc = nn.Linear(1024, num_classes)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=32, num_layers=1, batch_first=True)
        # self.lin2 = nn.Linear(1024, num_classes)
        # self.hidden = (torch.zeros(1, 1, 1024).cuda(), torch.zeros(1, 1, 1024).cuda())

    def forward(self, e_l, e_r, f):
        left = self.conv_left(e_l)
        right = self.conv_right(e_r)
        left = left.reshape(left.size(0), -1)
        right = right.reshape(right.size(0), -1)
        out = torch.cat((left, right), 1)
        f = f.reshape(f.size(0), -1)
        f = self.lin1(f)
        out = torch.cat((out, f), 1)
        out = self.lin2(out)
        out = self.fc(out)
        # out = out.view(sequence_length, -1, 1024)
        # out, self.hidden = self.lstm(out)
        # print(out.shape)
        # print(self.hidden[0].shape)
        # print(self.hidden[1].shape)
        # out = out.reshape(-1, sequence_length, 256)
        # out, self.hidden = self.lstm(out, self.hidden)
        # out = self.lin2(out)
        return out


def train_model():
    train_dataset = Dataset()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = ConvNet(num_classes).to(device)

    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("TRAIN...")
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (eye_left, eye_right, face, pos) in enumerate(train_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            outputs = model(eye_left, eye_right, face)
            loss = criterion(outputs, pos)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './model.pth')

    return model

'''
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
'''
