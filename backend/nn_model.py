import json
import time
from contextlib import suppress
import redis
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from backend.conf import settings


cache = redis.Redis(settings.redis_host, settings.redis_port, 0)


def get_task_state(task_id):
    result = cache.get(f'task_{task_id}')
    if result:
        result = json.loads(result)
    return result


def set_task_state(task_id, data):
    cache.set(f'task_{task_id}', json.dumps(data))


mlflow.set_tracking_uri(settings.ml_flow_server_url)
mlflow_client = mlflow.tracking.MlflowClient(settings.ml_flow_server_url)

seq_length = 50  # The sentence window size
step = 1  # The steps between the windows
production_stage = 'production'


class SimpleLSTM(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, dropout=0.2, **kwargs):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=2)
        self.embeddings = nn.Embedding(n_vocab, embedding_dim)
        self.fc = nn.Linear(hidden_dim, n_vocab)
        self.kwargs = kwargs

    def forward(self, seq_in):
        embedded = self.embeddings(seq_in.t())
        lstm_out, _ = self.lstm(embedded)
        ht=lstm_out[-1]
        out = self.fc(ht)
        return out


def train_nn_model(artist, lyrics_str, n_epochs=1, task_id: str = ''):
    chars = sorted(list(set(lyrics_str)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    sentences = []
    next_chars = []

    for i in range(0, len(lyrics_str) - seq_length, step):
        sentences.append(lyrics_str[i: i + seq_length])
        next_chars.append(lyrics_str[i + seq_length])

    sentences = np.array(sentences)
    next_chars = np.array(next_chars)

    def getdata(sentences, next_chars):
        X = np.zeros((len(sentences), seq_length))
        y = np.zeros((len(sentences)))
        for i in range(len(sentences)):
            sentence = sentences[i]
            for t, char in enumerate(sentence):
                X[i, t] = char_to_int[char]
            y[i] = char_to_int[next_chars[i]]
        return X, y

    train_x, train_y = getdata(sentences, next_chars)
    X_train_tensor = torch.tensor(train_x, dtype=torch.long)
    Y_train_tensor = torch.tensor(train_y, dtype=torch.long)

    train = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128)

    model = SimpleLSTM(len(chars), 256, 256, char_to_int=char_to_int, int_to_char=int_to_char)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    avg_losses_f = []
    with mlflow.start_run():
        for epoch in range(n_epochs):
            start_time = time.time()
            model.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            avg_loss = 0.
            for i, (x_batch, y_batch) in enumerate(train_loader):
                y_pred = model(x_batch)

                loss = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            avg_losses_f.append(avg_loss)
            elapsed_time = time.time() - start_time
            set_task_state(task_id, {
                'name': artist,
                'epochs': n_epochs,
                'current_epoch': epoch + 1,
                'loss': avg_loss,
                'time': elapsed_time
            })
        with suppress(Exception):
            mlflow_client.create_registered_model(artist)
        mlflow.log_metric("loss", np.average(avg_losses_f))
        mlflow.pytorch.log_model(model, "models", registered_model_name=artist)  # noqa
        last_model = max(mlflow_client.get_registered_model(artist).latest_versions, key=lambda x: int(x.version))  # noqa
        mlflow_client.transition_model_version_stage(artist, last_model.version, production_stage)  # noqa


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def predict_lyrics(artist, start_text, length, variance=0.25):
    model = mlflow.pytorch.load_model(f'models:/{artist}/{production_stage}')
    int_to_char = model.kwargs['int_to_char']
    char_to_int = model.kwargs['char_to_int']

    start_text = ''.join(sym for sym in start_text if sym in char_to_int)[:seq_length]

    generated = ''
    original = start_text
    window = start_text

    for i in range(length):
        x = np.zeros((1, seq_length))
        for t, char in enumerate(window):
            x[0, t] = char_to_int[char]

        x_in = Variable(torch.LongTensor(x))
        pred = model(x_in)
        pred = np.array(F.softmax(pred, dim=1).data[0])
        next_index = sample(pred, variance)
        next_char = int_to_char[next_index]  # noqa

        generated += next_char
        window = window[1:] + next_char  # Update Window for next char predict

    return original + generated
