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
    dir_seq_len = 'lstm/train/' + str(seq_len)
    if not os.path.exists(dir_seq_len):
        os.makedirs(dir_seq_len)

    dirname = dir_seq_len + '/' + str(uuid.uuid4()) + '/'
    os.mkdir(dirname)
    # print(dirname)
    for i in range(seq_len):
        np.save(dirname + '%d.points' % i, points[i])
        misc.imsave(dirname + '%d_l.png' % i, eyes_left[i])
        misc.imsave(dirname + '%d_r.png' % i, eyes_right[i])
        np.save(dirname + '%d.face' % i, faces[i])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dirname='./lstm/train', seq_len=5):
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
            self.eye_left.append([])
            self.eye_right.append([])
            self.face.append([])
            self.points.append([])
            # print(curr)
            for i in range(seq_len):
                self.face[index].append(np.load(self.dirname + curr + '/%d.face.npy' % i))
                self.points[index].append(np.load(self.dirname + curr + '/%d.points.npy' % i))
                self.eye_left[index].append(misc.imread(self.dirname + curr + '/%d_l.png' % i).reshape((3, 32, 32)))
                self.eye_right[index].append(misc.imread(self.dirname + curr + '/%d_r.png' % i).reshape((3, 32, 32)))
            # print(self.points[index])
        print("END LOAD DATA")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.eye_left[index])).float(), \
               torch.from_numpy(np.array(self.eye_right[index])).float(), \
               torch.from_numpy(np.array(self.face[index])).float(), \
               torch.from_numpy(np.array(self.points[index])).float()

