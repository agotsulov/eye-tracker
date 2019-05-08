import torch
import train
import model
import dataset
import utility

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_seq_len = 60

models = [
    model.TwoEyes,
    model.TwoEyesSameLayer,
    model.TwoEyesLSTM,
]

models_seq_len = [
    [1, 2, 4, 8, 16, 32],
    [2],
    [2, 4, 8, 16, 32],
]

max_samples = [
    [2500, 5000, 999999],
    [99999],
    [2500, 5000, 999999],
]

model = utility.load_model('./models/TwoEyesLSTM/model_{}.pth'.format(4),
                           device,
                           model.TwoEyesLSTM(2, 4))

train.val_model(model, dataset.Dataset(dirname='./val/1', seq_len=4, val=True))

for model_index in range(len(models)):
    for seq_len in models_seq_len[model_index]:
        for max_sample in max_samples[model_index]:
            train_dataset = dataset.Dataset(seq_len=seq_len, dataset_seq_len=dataset_seq_len, max_samples=max_sample)
            model = train.train_model(models[model_index](2, seq_len).to(device), seq_len, train_dataset)

            test_dataset = dataset.Dataset(dirname='./test', seq_len=seq_len)
            train.test_model(model, test_dataset)


