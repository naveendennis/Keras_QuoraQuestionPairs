import keras.preprocessing.text
import numpy as np
import string
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
import pandas as pd
import pickle
import os.path
from sklearn.model_selection import train_test_split


np.random.seed(1337)  # for reproducibility


def remove_punctuation(s):
    s = ''.join([i.lower() for i in s if i not in frozenset(string.punctuation)])
    return s


def pre_processing(data_contents):
    feature1 = data_contents['question1'].apply(remove_punctuation)
    feature2 = data_contents['question2'].apply(remove_punctuation)
    features = feature1 + ' ~$#|#$~ ' + feature2
    features = features.iloc[:].values
    tk = keras.preprocessing.text.Tokenizer(num_words=10000, lower=True, split=" ")
    tk.fit_on_texts(features)
    features = tk.texts_to_sequences(features)
    features = sequence.pad_sequences(features, maxlen=max_len)
    return features


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_name = os.path.join(dir_path, 'data','_blstm_rnn_model_B200_adam_mse_')
    if not os.path.exists(model_name):
        filename = os.path.join(dir_path, 'data', 'feature_train_test')
        if not os.path.exists(filename):

            data_contents = pd.read_csv(os.path.join(dir_path, 'dataset','train.csv'), sep=', ', delimiter=',', header='infer', names=None)
            data_contents = data_contents.dropna()

            feature1 = data_contents['question1'].apply(remove_punctuation)
            feature2 = data_contents['question2'].apply(remove_punctuation)
            feature = feature1 + ' ~$#|#$~ ' + feature2
            label = data_contents['is_duplicate']

            feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=0.80)
            with open(filename, 'wb') as f:
                pickle.dump(feature_train, f)
                pickle.dump(feature_test, f)
                pickle.dump(label_train, f)
                pickle.dump(label_test, f)

        else:
            with open(filename, 'rb') as f:
                feature_train = pickle.load(f)
                feature_test = pickle.load(f)
                label_train = pickle.load(f)
                label_test = pickle.load(f)

        feature_train = feature_train.iloc[:].values
        label_train = label_train.iloc[:].values
        feature_test = feature_test.iloc[:].values
        label_test = label_test.iloc[:].values

        tk = keras.preprocessing.text.Tokenizer(num_words=10000, lower=True, split=" ")
        tk.fit_on_texts(feature_train)
        feature_train = tk.texts_to_sequences(feature_train)

        max_len = 1024
        print("max_len ", max_len)
        print('Pad sequences (samples x time)')

        tk = keras.preprocessing.text.Tokenizer(num_words=10000, lower=True, split=" ")
        tk.fit_on_texts(feature_test)
        feature_test = tk.texts_to_sequences(feature_test)
        feature_test = sequence.pad_sequences(feature_test, maxlen=max_len)

        feature_train = sequence.pad_sequences(feature_train, maxlen=max_len)
        lstm_size = 256
        max_features = 20000
        batch_size = 200

        model = Sequential()
        model.add(Embedding(max_features, lstm_size, input_length=max_len))
        model.add(Dense(512, activation='softmax'))
        model.add(Bidirectional(LSTM(lstm_size, recurrent_dropout=0.2)))
        model.add(Dense(1, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        model.fit(feature_train, y=label_train, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.2, shuffle=True)
        model.save(model_name)
    else:
        model = load_model(model_name)
    
    test_contents = pd.read_csv(os.path.join(dir_path, 'dataset','test.csv'), sep=', ', delimiter=',', header='infer', names=None)
    test_contents = test_contents.dropna()
    test_contents = pre_processing(test_contents)
    label_contents = model.predict(test_contents, batch_size=batch_size)
    with open(os.path.join(dir_path, '_label_test')) as f:
        pickle.dump(label_contents, f)
    scores = model.evaluate(feature_test, label_test, verbose=1)
    print(scores)
