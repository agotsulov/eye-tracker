import rnn.train

models_seq_len = [1, 2, 4, 6, 8, 10, 16, 32]

for i in models_seq_len:
    print("Current model seq_len:{}".format(i))
    model = rnn.train.train_model(i)
    rnn.train.test_model(model, i)


