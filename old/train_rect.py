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
from math import ceil


class Dataset(torch.utils.data.Dataset):
    def __init__(self, screen_w, screen_h, num_rects=4):
        self.heatmap = np.zeros(num_rects * num_rects)
        self.len = len(os.listdir('data'))
        print(self.len)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.eyes = np.ndarray((self.len, 3, 64, 32))
        self.face = np.ndarray((self.len, 68, 2))
        self.x = [0 for i in range(self.len)]
        self.y = [0 for i in range(self.len)]
        self.result = np.zeros((self.len, num_rects * num_rects))

        print("LOADING DATA...")
        names = os.listdir('data')
        for index in range(self.len):
            curr = names[index]
            frame = misc.imread('data/' + curr)

            rects = detector(frame, 0)

            shape = None
            eyes_ = None

            if rects is None:
                if os.path.exists(curr):
                    os.remove(curr)
                print('NOT FOUND RECT: ' + curr)
                self.len -= 1
                print(self.len)
                continue

            for (i, rect) in enumerate(rects):
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:42]]))
                eye_left = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

                (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[43:48]]))
                eye_right = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

                if eye_left is not None and eye_right is not None:
                    eyes_ = np.concatenate((cv2.resize(eye_left, (32, 32)), cv2.resize(eye_right, (32, 32))), axis=1)

            if eyes_ is None:
                if os.path.exists(curr):
                    os.remove(curr)
                print('NOT FOUND EYES: ' + curr)
                self.len -= 1
                print(self.len)
                continue

            self.face[index] = shape

            x = ceil(int(curr.split('_')[2]) / (screen_w / num_rects))
            y = ceil(int(curr[:-4:].split('_')[4]) / (screen_h / num_rects))

            # print(x)
            # print(y)

            self.result[index][(y - 1) * num_rects + (x - 1)] = 1

            self.heatmap[(y - 1) * num_rects + (x - 1)] += 1

            self.eyes[index] = eyes_.reshape((3, 64, 32))

            # print(self.result[index])
        print(self. heatmap)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(self.eyes[index]).float(),\
               torch.from_numpy(self.face[index]).float(),\
               torch.from_numpy(self.result[index]).float()


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            # nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            # nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            # nn.Dropout2d(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(1024),
            # nn.Dropout2d(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * 4 * 2 + 68 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            # nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            # nn.Dropout(0.1),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, e, f):
        out = self.conv1(e)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        f = f.reshape(f.size(0), -1)
        out = torch.cat((out, f), 1)
        out = self.fc(out)
        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
num_classes = 16
batch_size = 50
learning_rate = 0.001


def train_model():
    train_dataset = Dataset(1920, 1080)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    # criterion = torch.nn.MultiLabelSoftMarginLoss()
    # criterion = torch.nn.MSELoss(reduction='sum')
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("TRAIN...")
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (eyes, face, pos) in enumerate(train_loader):
            eyes = eyes.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            outputs = model(eyes, face)
            # print(outputs)
            # print(pos)
            # pos = pos.squeeze_()
            # print(pos)
            loss = criterion(outputs, pos)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './model_rect.pth')

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
# Save the model checkpoint
