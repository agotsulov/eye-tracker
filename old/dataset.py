import numpy as np
import os
import torch.utils.data.dataset
import torch
from imutils import face_utils
import dlib
import cv2
import os
from scipy import misc
import pygame
import time


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dirname='./data'):
        self.eyes = []
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

            eyes_ = None
            shape = None

            if rects is None:
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

            self.add(int(curr.split('_')[2]), int(curr[:-4:].split('_')[4]), eyes_.reshape((3, 64, 32)), shape)
        print("END LOAD DATA")

    def add(self, x, y, eyes, face):
        self.x.append(x)
        self.y.append(y)
        self.eyes.append(eyes)
        self.face.append(face)
        self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.from_numpy(self.eyes[index]).float(), \
               torch.from_numpy(self.face[index]).float(), \
               torch.from_numpy(np.array([self.x[index], self.y[index]])).float()

