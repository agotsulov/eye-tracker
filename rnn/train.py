import torch
import rnn.model
import rnn.dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 50
num_classes = 2
batch_size = 10
sequence_length = 16
learning_rate = 0.001


def train_model(seq_len):
    train_dataset = rnn.dataset.Dataset(seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = rnn.model.TwoEyes(num_classes, batch_size, seq_len).to(device)

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
            loss = criterion(outputs, pos[:, -1, :])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './rnn/models/model_{}.pth'.format(seq_len))

    return model


def test_model(model, seq_len):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    criterion = torch.nn.MSELoss().cuda()

    test_dataset = rnn.dataset.Dataset(dirname='rnn/test', seq_len=seq_len)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    min_loss = 999999

    with torch.no_grad():
        for i, (eye_left, eye_right, face, pos) in enumerate(test_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            outputs = model(eye_left, eye_right, face)
            loss = criterion(outputs, pos[:, -1, :])

            if loss.item() < min_loss:
                min_loss = loss.item()

            print('Loss: {:.4f}'.format(loss.item()))

        print('MIN LOSS: {} '.format(min_loss))
        print('TOTAL LOSS: {} '.format(loss.item()))

