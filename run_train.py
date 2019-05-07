import torch
import train
import model
import utility


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

models_seq_len = [32]

models = [
    model.TwoEyes,
    # rnn.model.TwoEyesSameLayer
]

model = utility.load_model('./models/TwoEyes/model_2.pth', device, model.TwoEyes(2, 2))
train.test_model(model, 2)

for i in models_seq_len:
    for m in models:
        model = train.train_model(m(2, i).to(device), i)
        train.test_model(model, i)


