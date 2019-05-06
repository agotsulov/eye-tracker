import torch
from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
from scipy import misc
import torchvision
import torchvision.transforms as transforms
import time
import uuid


def save_data(seq_len, eyes_left, eyes_right, faces, points):
    dir_seq_len = 'rnn/train/' + str(seq_len)
    if not os.path.exists(dir_seq_len):
        os.makedirs(dir_seq_len)

    dirname = dir_seq_len + '/' + str(uuid.uuid4()) + '/'
    os.mkdir(dirname)

    eyes_left = np.array(eyes_left)
    eyes_right = np.array(eyes_right)
    faces = np.array(faces)
    points = np.array(points)

    eyes_left = eyes_left.transpose((0, 3, 1, 2)).reshape((3 * seq_len, 32, 32))
    eyes_right = eyes_right.transpose((0, 3, 1, 2)).reshape((3 * seq_len, 32, 32))

    np.save(dirname + 'points', points)
    np.save(dirname + 'eye_left', eyes_left)
    np.save(dirname + 'eye_right', eyes_right)
    np.save(dirname + 'faces', faces)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dirname='./rnn/train', seq_len=5):
        self.eye_left = []
        self.eye_right = []
        self.face = []
        self.points = []
        self.dirname = dirname + '/' + str(seq_len) + '/'
        self.size = len(os.listdir(self.dirname))

        names = os.listdir(self.dirname)

        print(self.size)
        print("LOADING DATA...")

        for index in range(len(os.listdir(self.dirname))):
            curr = names[index]
            # print(curr)
            self.face.append(np.load(self.dirname + curr + '/faces.npy'))
            self.points.append(np.load(self.dirname + curr + '/points.npy'))
            self.eye_left.append(np.load(self.dirname + curr + '/eye_left.npy'))
            self.eye_right.append(np.load(self.dirname + curr + '/eye_right.npy'))
            # print(self.points[index])
        print("END LOAD DATA")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.eye_left[index])).float(), \
               torch.from_numpy(np.array(self.eye_right[index])).float(), \
               torch.from_numpy(np.array(self.face[index])).float(), \
               torch.from_numpy(np.array(self.points[index])).float()

