from sklearn import metrics
from sklearn.metrics import classification_report

from sentence_classification import *

import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, Embedding
from keras.layers import Conv1D, GlobalMaxPool1D, MaxPool1D, LSTM, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_words = 2 ** 12
batch_size = 512
max_len = 32


# In[15]:

def cnn_clf(X_train, y_train, X_test, y_test):
    train = [" ".join(x["lemmas"]) for x in X_train]
    test = [" ".join(x["lemmas"]) for x in X_test]

    # In[16]:

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)

    # In[17]:

    x_train = tokenizer.texts_to_sequences(train)
    x_test = tokenizer.texts_to_sequences(test)

    # In[18]:

    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)

    num_classes = 5
    y_train = keras.utils.to_categorical([y - 1 for y in y_train], num_classes)
    y_val = keras.utils.to_categorical([y - 1 for y in y_test], num_classes)

    main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
    emb = Embedding(output_dim=128, input_dim=max_words, input_length=max_len)(main_input)
    emb = SpatialDropout1D(0.1)(emb)

    # tower_1 = Conv1D(64, 1, padding='same', activation='relu')(emb)
    tower_1 = Conv1D(256, 3, padding='valid', activation='relu')(emb)
    # tower_1 = LSTM(128)(tower_1)
    tower_1 = SpatialDropout1D(0.1)(tower_1)
    tower_1 = GlobalMaxPool1D()(tower_1)

    # tower_2 = Conv1D(64, 1, padding='same', activation='relu')(emb)
    tower_2 = Conv1D(256, 5, padding='valid', activation='relu')(emb)
    # tower_2 = LSTM(128)(tower_2)
    tower_2 = SpatialDropout1D(0.1)(tower_2)
    tower_2 = GlobalMaxPool1D()(tower_2)

    # tower_3 = MaxPool1D(3, padding='same')(emb)
    tower_3 = Conv1D(256, 7, padding='valid', activation='relu')(emb)
    # tower_3 = LSTM(128)(tower_3)
    tower_3 = SpatialDropout1D(0.1)(tower_3)
    tower_3 = GlobalMaxPool1D()(tower_3)

    # aux_input = Input(shape=(50,), dtype='float', name='aux_input')
    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    # output = Dropout(0.1)(output)
    # output = keras.layers.concatenate([output, aux_input], axis=1)
    sm = Dense(num_classes, activation='softmax')(output)

    model = Model(inputs=main_input, outputs=sm)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])

    epochs = 10

    # In[40]:

    from keras.callbacks import TensorBoard
    tensorboard = TensorBoard(log_dir='./logs', write_graph=True, write_images=True)
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[tensorboard, early_stopping])

    results = model.predict(x_test, batch_size=batch_size, verbose=1)
    return results


if __name__ == "__main__":
    predicted = []
    gold = []
    f_measures = []
    for i in range(1, 6):
        f_train = "data/folds/" + str(i) + "/train.txt"
        f_test = "data/folds/" + str(i) + "/test.txt"
        X_train, y_train = load_data(f_train)
        X_test, y_test = load_data(f_test)
        predicted_fold = cnn_clf(X_train, y_train, X_test, y_test)
        predicted_fold = np.round(predicted_fold)
        predicted.extend(predicted_fold)
        gold.extend(y_test)
        f_measures.append(metrics.f1_score(y_test, predicted_fold, average='macro'))
    print f_measures
    print classification_report(gold, predicted, digits=3)
    print metrics.precision_score(gold, predicted, average='macro')
    print metrics.recall_score(gold, predicted, average='macro')
    print metrics.f1_score(gold, predicted, average='macro')
