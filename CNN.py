import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def normalize(df):
    TAG_MIN = df[df.columns].min()
    TAG_MAX = df[df.columns].max()
    ndf = df.copy()
    for c in df.columns:
        try:
            if TAG_MIN[c] == TAG_MAX[c]:
                ndf[c] = df[c] - TAG_MIN[c]
            else:
                ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
        except:
            continue
    return ndf

if __name__ == "__main__":
    g = glob('sample/*')
    df = pd.read_csv(g[2])

    print(df.columns)

    df = df.drop(df.std()[df.std() < .3].index.values, axis=1)
    df = df.drop(df.std()[df.std() > 1000].index.values, axis=1)

    df.replace(to_replace="Benign",         value=0, inplace=True)
    df.replace(to_replace="FTP-BruteForce", value=1, inplace=True)
    df.replace(to_replace="SSH-Bruteforce", value=1, inplace=True)

    df.drop('Timestamp', axis=1, inplace=True)
    df = normalize(df).dropna()

    x = df[df.columns].drop('Label', axis=1).dropna()

    #print(x.columns)

    y = df[['Label']]

    print(x.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    print('Train images shape:', X_train.shape)
    print('Train labels shape:', y_train.shape)
    print('Test images shape:', X_test.shape)
    print('Test labels shape:', y_test.shape)
    print('Train labels:', y_train)
    print('Test labels:', y_test)

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=28))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(lr=0.01)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=20, verbose=1, batch_size=100)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    history_dict = history.history
    history_dict.keys()
    acc = history_dict['accuracy']
    loss = history_dict['loss']
    epochs = range(1, len(acc) + 1)

    #plt.plot(epochs, loss, 'bo', label='train loss')
    #plt.title('Train and val loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('loss')
    #plt.legend()
    #plt.show()
#
    #plt.clf()  # clear figure
    #plt.plot(epochs, acc, 'bo', label='Training acc')
    #plt.title('Training and validation accuracy')
    #plt.xlabel('Epochs')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()

    testdata = pd.read_csv(g[0])
    print(testdata.head(3))

    predictions = model.predict(X_test)
    answer = []
    for prediction in (predictions):
        print(np.argmax(prediction))