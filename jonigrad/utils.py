import numpy as np
from keras.datasets import mnist
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset


def compute_accuracy(model, X_test, y_test):
    correct_predictions = 0
    total_predictions = X_test.shape[0]
    for i in range(total_predictions):
        Xb = X_test[i : i + 1]
        out = model(Xb)
        prediction = np.argmax(out, axis=1)
        if prediction == y_test[i]:
            correct_predictions += 1
    return correct_predictions / total_predictions


def load_mnist(flatten=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    else:
        WIDTH, HEIGHT = X_train.shape[1], X_train.shape[2]
        X_train = X_train.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0
        X_test = X_test.reshape(-1, 1, HEIGHT, WIDTH).astype(np.float32) / 255.0

    return X_train, y_train, X_test, y_test


def load_temperature_data(seq_length):
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    data = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
    temperatures = data["Temp"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_temperatures = scaler.fit_transform(temperatures)

    n_test_years = 3
    test_size = n_test_years * 365  # number of days in 3 years
    train_data = normalized_temperatures[:-test_size]
    test_data = normalized_temperatures[-test_size:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    return X_train, y_train, X_test, y_test, data, temperatures, scaler




def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def build_vocab(data):
    special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    vocab = dict()
    # Initialize vocabularies with special tokens
    for token in special_tokens:
        vocab[token] = len(vocab)

    for sentence in data:
        for word in sentence.split():
            word = word.lower().replace(".", "").replace(",","").replace('"', "").replace(':', "")
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def tokenize(vocab, data):
    tok_data = []

    for sentence in data:
        sentence_tok = []
        sentence_tok.append(vocab["<SOS>"])
        for word in sentence.split():
            word = word.lower().replace(".", "").replace(",","").replace('"', "").replace(':', "")
            if word in vocab:
                sentence_tok.append(vocab[word])
            else:
                sentence_tok.append(vocab["<UNK>"])
        sentence_tok.append(vocab["<EOS>"])

        tok_data.append(sentence_tok)

    return tok_data


def pad_sentences(data, pad_token):
    max_len = max(len(sentence) for sentence in data)
    padded_data = []
    for sentence in data:
        while len(sentence) < max_len:
            sentence.append(pad_token)
        padded_data.append(sentence)
    return np.array(padded_data)



def load_fi_en_translations(debug=False):
    dataset = load_dataset("opus_books", "en-fi")["train"]

    finnish_data = []
    english_data = []
    if debug:
        for i in range(20):
            finnish_data.append(dataset[i]["translation"]["fi"])
            english_data.append(dataset[i]["translation"]["en"])
    else:
        for example in dataset:
            finnish_data.append(example["translation"]["fi"])
            english_data.append(example["translation"]["en"])
    
    dataset = load_dataset("EkBass/fin-eng-dataset")["train"]

    if debug:
        for i in range(20):
            finnish_data.append(dataset[i]["Finnish"])
            english_data.append(dataset[i]["English"])
    else:
        for i in range(len(dataset)):
            finnish_data.append(dataset[i]["Finnish"])
            english_data.append(dataset[i]["English"])

    en_vocab, fi_vocab = build_vocab(english_data), build_vocab(finnish_data)
    en_data_tok, fi_data_tok = tokenize(en_vocab, english_data), tokenize(
        fi_vocab, finnish_data
    )

    en_data = pad_sentences(en_data_tok, en_vocab["<PAD>"])
    fi_data = pad_sentences(fi_data_tok, fi_vocab["<PAD>"])

    print(f"Finnish vocab {len(fi_vocab)}, English vocab {len(en_vocab)}")
    print(f"Finnish seq len {fi_data.shape[1]}, English seq len {en_data.shape[1]}")
    print(f"Sentences {fi_data.shape[0]}")

    if np.any(en_data < 0):
                print("Negative indices found in en_data:")
                print(en_data)
    if np.any(fi_data < 0):
        print("Negative indices found in fi_data:")
        print(fi_data)
    return en_data, en_vocab, fi_data, fi_vocab

if __name__ == '__main__':
    load_fi_en_translations() 