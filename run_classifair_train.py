import model
import torch
import dataset
import model
import math
import numpy as np
import utility
import os
import logging


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_seq_len = 60

log = logging.getLogger("ctrain")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler("logs/train.log")
log_file_handler.setFormatter(formatter)
log.addHandler(log_file_handler)
log_console_handler = logging.StreamHandler()
log.addHandler(log_console_handler)
log.setLevel(logging.INFO)

test_log = logging.getLogger("ctest")
test_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
test_log_file_handler = logging.FileHandler("logs/train.log")
test_log_file_handler.setFormatter(formatter)
test_log.addHandler(test_log_file_handler)
test_log_file_handler = logging.FileHandler("logs/test.log")
test_log_file_handler.setFormatter(formatter)
test_log.addHandler(test_log_file_handler)
test_log_console_handler = logging.StreamHandler()
test_log.addHandler(test_log_console_handler)
test_log.setLevel(logging.INFO)

val_log = logging.getLogger("cval")
val_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
val_log_file_handler = logging.FileHandler("logs/val.log")
val_log_file_handler.setFormatter(formatter)
val_log.addHandler(val_log_file_handler)
val_log_console_handler = logging.StreamHandler()
val_log.addHandler(val_log_console_handler)
val_log.setLevel(logging.INFO)


def train_model(model, seq_len, train_dataset, num_rects, batch_size=50, num_epochs=50, learning_rate=0.001):

    log.info("CURRENT MODEL seq_len: {}".format(seq_len))
    log.info("CURRENT MODEL num_rects: {}".format(num_rects))
    log.info("CURRENT MODEL: {}".format(model.__class__.__name__))
    log.info("CURRENT DATASET SIZE: {}".format(train_dataset.__len__()))

    print(batch_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    criterion = torch.nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log.info("TRAIN...")

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (eye_left, eye_right, face, pos) in enumerate(train_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            outputs = model(eye_left, eye_right, face)
            # print(outputs.size())
            # print(pos.size())
            # print(pos.size())
            loss = criterion(outputs, pos)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}'
                         .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    dirname = './models/{}/'.format(model.__class__.__name__)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(model.state_dict(), dirname + 'model_{}.pth'.format(seq_len))
    torch.save(model.state_dict(), dirname + 'model_{}_{}_{}.pth'.format(train_dataset.__len__(), num_rects, seq_len))
    log.info('Save new model: ' + dirname + 'model_{}_{}_{}.pth'.format(train_dataset.__len__(), num_rects, seq_len))

    return model


def test_model(model, test_dataset, batch_size=50):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

    criterion = torch.nn.BCELoss().cuda()

    test_log.info("CURRENT MODEL seq_len: {}".format(test_dataset.seq_len))
    test_log.info("CURRENT MODEL: {}".format(model.__class__.__name__))
    test_log.info("CURRENT DATASET SIZE: {}".format(test_dataset.__len__()))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    correct = 0
    total = 0

    min_loss = 999999
    max_loss = 0
    sum_loss = 0
    count_loss = 0

    with torch.no_grad():
        for i, (eye_left, eye_right, face, pos) in enumerate(test_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            out = model(eye_left, eye_right, face)
            loss = criterion(out, pos)

            _, predicted = torch.max(out.data, 1)
            _, pos = torch.max(pos.data, 1)
            total += pos.size(0)
            # print(predicted)
            # print(pos)
            correct += (predicted == pos).sum().item()
            if loss.item() < min_loss:
                min_loss = loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()
            sum_loss += loss.item()
            count_loss += 1

            # log.info('MSELoss: {:.4f}'.format(loss.item()))

        test_log.info('MIN BCELoss: {} '.format(min_loss))
        test_log.info('MAX BCELoss: {} '.format(max_loss))
        test_log.info('AVG BCELoss: {}'.format(sum_loss / count_loss))
    print('Accuracy : %d %%' % (100 * correct / total))
    test_log.info('Accuracy : %d %%' % (100 * correct / total))
    return test_dataset.__len__()


# model = model.EyeClassifier(4, 2).cuda()

#train_dataset = dataset.CDataset(num_rects=2, seq_len=2, dataset_seq_len=dataset_seq_len, max_samples=2500)
#model = train_model(model, 2, train_dataset)

def train_test(num_rects, w, seq_len, max_samples, lstm=False, batch_size=50):
    if lstm is False:
        model_ = model.EyeClassifier(num_rects, seq_len).cuda()
    else:
        model_ = model.EyeClassifierLSTM(num_rects, seq_len).cuda()

    train_dataset = dataset.CDataset(num_rects=w, seq_len=seq_len, dataset_seq_len=dataset_seq_len,
                                     max_samples=max_samples)
    model_ = train_model(model_, seq_len, train_dataset, num_rects=num_rects, batch_size=batch_size)

    test_dataset = dataset.CDataset(num_rects=w, dirname='./test', seq_len=seq_len)
    test_model(model_, test_dataset, batch_size=batch_size)


# train_test(4, 2, 2, 2500, lstm=True, batch_size=35)
# train_test(4, 2, 2, 5000, lstm=True, batch_size=35)
# train_test(4, 2, 2, 999999, lstm=True, batch_size=35)
# train_test(16, 4, 2, 2500, lstm=True, batch_size=35)
# train_test(16, 4, 2, 5000, lstm=True, batch_size=35)
# train_test(16, 4, 2, 999999, lstm=True, batch_size=35)

# train_test(4, 2, 4, 2500, lstm=True, batch_size=35)
# train_test(4, 2, 4, 5000, lstm=True, batch_size=35)
# train_test(4, 2, 4, 999999, lstm=True, batch_size=35)
# train_test(16, 4, 4, 2500, lstm=True, batch_size=35)
# train_test(16, 4, 4, 5000, lstm=True, batch_size=35)
# train_test(16, 4, 4, 999999, lstm=True, batch_size=35)

# train_test(4, 2, 8, 2500, lstm=True, batch_size=20)
# train_test(4, 2, 8, 5000, lstm=True, batch_size=20)
# train_test(4, 2, 8, 999999, lstm=True, batch_size=20)
# train_test(16, 4, 8, 2500, lstm=True, batch_size=20)

# train_test(64, 8, 4, 2500, lstm=True, batch_size=20)
# train_test(64, 8, 4, 5000, lstm=True, batch_size=20)
# train_test(64, 8, 4, 999999, lstm=True, batch_size=20)

# train_test(64, 8, 8, 2500, lstm=True, batch_size=20)
# train_test(64, 8, 8, 5000, lstm=True, batch_size=20)
# train_test(64, 8, 8, 999999, lstm=True, batch_size=20)

# train_test(16, 4, 8, 5000, lstm=True, batch_size=20)
# train_test(16, 4, 8, 999999, lstm=True, batch_size=20)

train_test(256, 16, 8, 999999, lstm=True, batch_size=20)
train_test(256, 16, 4, 999999, lstm=True, batch_size=20)
train_test(256, 16, 2, 999999, lstm=True, batch_size=20)
train_test(256, 16, 8, 2500, lstm=True, batch_size=20)
train_test(256, 16, 8, 5000, lstm=True, batch_size=20)

train_test(4, 2, 1, 2500)
train_test(4, 2, 1, 5000)
train_test(4, 2, 1, 999999)
train_test(16, 4, 1, 2500)
train_test(16, 4, 1, 5000)
train_test(16, 4, 1, 999999)

train_test(4, 2, 2, 2500)
train_test(4, 2, 2, 5000)
train_test(4, 2, 2, 999999)
train_test(16, 4, 2, 2500)
train_test(16, 4, 2, 5000)
train_test(16, 4, 2, 999999)



