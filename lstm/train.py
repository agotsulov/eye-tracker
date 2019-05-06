import torch
import rnn.model
import rnn.dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
num_classes = 2
batch_size = 25
sequence_length = 5
learning_rate = 0.001


def train_model():
    train_dataset = rnn.dataset.Dataset(seq_len=sequence_length)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = rnn.model.TwoEyesWithLSTM(num_classes, batch_size, sequence_length).to(device)

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
            if epoch >= 9 or epoch == 1:
                print(outputs)
                print(pos[:, -1, :])
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './lstm/model.pth')

    return model


def test_model(model):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    criterion = torch.nn.MSELoss().cuda()

    test_dataset = rnn.dataset.Dataset(dirname='lstm/test', seq_len=sequence_length)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    with torch.no_grad():
        for i, (eye_left, eye_right, face, pos) in enumerate(test_loader):
            eye_left = eye_left.to(device)
            eye_right = eye_right.to(device)
            face = face.to(device)
            pos = pos.to(device)

            # Forward pass
            outputs = model(eye_left, eye_right, face)
            loss = criterion(outputs, pos[:, -1, :])

            print(outputs)
            print(pos[:, -1, :])

            print('Loss: {:.4f}'.format(loss.item()))

        print('TOTAL LOSS: {} '.format(loss.item()))

