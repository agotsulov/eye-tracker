import torch
import dataset
import math
import logging

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
num_classes = 2
batch_size = 50
sequence_length = 16
learning_rate = 0.001

log = logging.getLogger("train")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler("train.log")
log_file_handler.setFormatter(formatter)
log.addHandler(log_file_handler)
log_console_handler = logging.StreamHandler()
log.addHandler(log_console_handler)
log.setLevel(logging.INFO)


def train_model(model, seq_len, dataset_seq_len=60):
    log.info("CURRENT MODEL seq_len: {}".format(seq_len))
    log.info("CURRENT MODEL: {}".format(model.__class__.__name__))

    train_dataset = dataset.Dataset(seq_len=seq_len, dataset_seq_len=dataset_seq_len)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    criterion = torch.nn.MSELoss().cuda()
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

            loss = criterion(outputs, pos[:, -1, :])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 25 == 0:
                log.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                         .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './models/{}/model_{}.pth'.format(model.__class__.__name__, seq_len))



    return model


def test_model(model, seq_len):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    criterion = torch.nn.MSELoss().cuda()

    test_dataset = dataset.Dataset(dirname='./test', seq_len=seq_len)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    min_loss = 999999

    sum_loss = 0
    count_loss = 0

    max_loss = 0

    with torch.no_grad():
        for i, (eye_left, eye_right, face, pos) in enumerate(test_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            out = model(eye_left, eye_right, face)
            loss = criterion(out, pos[:, -1, :])

            if loss.item() < min_loss:
                min_loss = loss.item()

            if loss.item() > max_loss:
                max_loss = loss.item()

            sum_loss += loss.item()
            count_loss += 1

            log.info('Loss: {:.4f}'.format(loss.item()))

        log.info('MIN LOSS: {} '.format(min_loss))
        log.info('TOTAL LOSS: {} '.format(loss.item()))
        log.info('AVG LOSS: {}'.format(sum_loss / count_loss))
