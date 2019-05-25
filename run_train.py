import torch
import train
import model
import dataset
import pandas as pd
import utility
import seaborn as sns
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_seq_len = 60


def train_test(num_rects_, seq_len_, max_samples_):
    model_ = model.EyesLSTMWithClassifier(num_classes=2, num_rects=num_rects_, seq_len=seq_len_).cuda()

    train_dataset_ = dataset.Dataset(seq_len=seq_len_, dataset_seq_len=dataset_seq_len, max_samples=max_samples_)
    model_ = train.train_model(model_, seq_len_, train_dataset_)

    test_dataset_ = dataset.Dataset(dirname='./test', seq_len=seq_len_)
    train.test_model(model_, test_dataset_, num_rects=num_rects_)


train_test(64, 4, 2500)
train_test(256, 8, 10770)
train_test(64, 8, 10770)
train_test(16, 8, 10770)
train_test(4, 8, 10770)







models = [
    # model.TwoEyes,
    # model.TwoEyesSameLayer,
    # model.TwoEyesRNN,
    # model.TwoEyesGRU,
    # model.TwoEyesLSTM,
    # model.TwoEyes,
    model.TwoEyesLSTM,
    # model.TwoEyesRNN,
    # model.TwoEyesGRU,
]

models_seq_len = [
    # [1, 2, 4, 8],
    # [2],
    # [2, 4, 8],
    # [8], # [2, 4, 8],
    # [2, 4, 8],
    # [1, 2, 4, 8],
    # [2, 4, 8],
    [8]
    # [8],
    # [2, 4, 8],
]

max_samples = [
    # [2500, 5000 ],
    # [2500, 5000],
    # [2500, 5000],
    # [5000],
    # [2500, 5000],
    # [99999],
    [99999],
    # [99999],
    # [99999],
]

results = pd.DataFrame(columns=['model',
                                'train size',
                                'test dataset',
                                'test size',
                                'min error',
                                'max error',
                                'avg error',
                                'median error',
                                'errors'])

for model_index in range(len(models)):
    for seq_len in models_seq_len[model_index]:
        for max_sample in max_samples[model_index]:

            train_dataset = dataset.Dataset(seq_len=seq_len, dataset_seq_len=dataset_seq_len, max_samples=max_sample)
            model = train.train_model(models[model_index](2, seq_len).to(device), seq_len, train_dataset)

            model_name = model.__class__.__name__
            train_size = train_dataset.__len__()

            t_name = './test'
            test_dataset = dataset.Dataset(dirname='./test', seq_len=seq_len)
            t_size,  min_e, max_e, avg_e, m_e, es = \
                train.test_model(model, test_dataset)
            results = results.append({'model': model_name,
                                      'train size': train_size,
                                      'test dataset': t_name,
                                      'test size': t_size,
                                      'min error': min_e,
                                      'max error': max_e,
                                      'avg error': avg_e,
                                      'median error': m_e,
                                      'errors': es
                                      }, ignore_index=True)
            
            t_name = './val/1'
            t_size,  min_e, max_e, avg_e, m_e, es = \
                train.val_model(model, dataset.Dataset(dirname='./val/1', seq_len=seq_len, val=True))
            results = results.append({'model': model_name,
                                      'train size': train_size,
                                      'test dataset': t_name,
                                      'test size': t_size,
                                      'min error': min_e,
                                      'max error': max_e,
                                      'avg error': avg_e,
                                      'median error': m_e,
                                      'errors': es
                                      }, ignore_index=True)

            t_name = './val/2'
            t_size, min_e, max_e, avg_e, m_e, es = \
                train.val_model(model, dataset.Dataset(dirname='./val/2', seq_len=seq_len, val=True))
            results = results.append({'model': model_name,
                                      'train size': train_size,
                                      'test dataset': t_name,
                                      'test size': t_size,
                                      'min error': min_e,
                                      'max error': max_e,
                                      'avg error': avg_e,
                                      'median error': m_e,
                                      'errors': es
                                      }, ignore_index=True)

            t_name = './val/3'
            t_size, min_e, max_e, avg_e, m_e, es = \
                train.val_model(model, dataset.Dataset(dirname='./val/3', seq_len=seq_len, val=True))
            results = results.append({'model': model_name,
                                      'train size': train_size,
                                      'test dataset': t_name,
                                      'test size': t_size,
                                      'min error': min_e,
                                      'max error': max_e,
                                      'avg error': avg_e,
                                      'median error': m_e,
                                      'errors': es
                                      }, ignore_index=True)

            t_name = './val/4'
            t_size, min_e, max_e, avg_e, m_e, es = \
                train.val_model(model, dataset.Dataset(dirname='./val/4', seq_len=seq_len, val=True))
            results = results.append({'model': model_name,
                                      'train size': train_size,
                                      'test dataset': t_name,
                                      'test size': t_size,
                                      'min error': min_e,
                                      'max error': max_e,
                                      'avg error': avg_e,
                                      'median error': m_e,
                                      'errors': es
                                      }, ignore_index=True)

            results_file = open('results.log', 'a')
            results_to_csv = results.to_csv(index=False)
            results_file.write(results_to_csv)
            print(results_to_csv)
            results_file.close()

