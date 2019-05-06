import os
import torch
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math
import time
from scipy.misc import imsave


def load_model(filename, device, model_):
    model = None
    model_file = os.path.exists(filename)
    if model_file:
        print("FOUND MODEL")
        model = model_.to(device)
        model.load_state_dict(torch.load(filename, map_location='cuda:0' if torch.cuda.is_available() else 'cpu'))
        model.eval()
    else:
        print("MODEL NOT FOUND")
    return model




def predict(eyes, shape, model, device, max_w=1920, max_h=1080):
    if eyes is not None and model is not None:
        eyes = eyes.reshape((1, 3, 64, 32))
        shape = shape.reshape((1, 68, 2))

        eyes_torch = torch.from_numpy(eyes).float().to(device)
        shape_torch = torch.from_numpy(shape).float().to(device)
        out = model(eyes_torch, shape_torch)

        out = out.cpu().data.numpy()[0]

        x_pred = out[0]
        y_pred = out[1]

        if x_pred < 0:
            x_pred = 0
        if y_pred < 0:
            y_pred = 0
        if x_pred > max_w:
            x_pred = max_w
        if y_pred > max_h:
            y_pred = max_h

        return x_pred, y_pred
    return None





def uncut(pred, num_rects, screen_w, screen_h):

    size = math.sqrt(num_rects)

    w = int(screen_w / size)
    h = int(screen_h / size)

    i = pred.argmax()

    # print(pred)
    # print(i)

    y = int(i / size) * h

    x = int((i - int(i / size)) / size) * w

    return x, y, w, h


class FaceDetector:
    def __init__(self, predictor_filename="shape_predictor_68_face_landmarks.dat"):
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_filename)
        if self.predictor is None:
            print("PREDICTOR NOT FOUND")

    def predict(self, frame):
        eyes = None
        shape = None

        rects = self.detector(frame, 0)

        for (i, rect) in enumerate(rects):
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:42]]))
            eye_left = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

            (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[43:48]]))
            eye_right = frame[y_ - 3:y_ + h_ + 3, x_ - 5:x_ + w_ + 5]

            if eye_left is not None and eye_right is not None:
                eyes = np.concatenate((cv2.resize(eye_left, (32, 32)), cv2.resize(eye_right, (32, 32))), axis=1)

            if eyes is not None:
                eyes = cv2.resize(eyes, (64, 32))

        return eyes, shape
