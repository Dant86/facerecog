from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from sklearn.utils import shuffle
from PIL import Image
import numpy as np
import os

PHOTO_DIR = "photos/"
PHOTO_ROWS, PHOTO_COLS = 28, 28

MODEL_DIR = "model/"

class FaceRecogPredictor:

    def __init__(self, labels, from_json=False):
        self.from_json = from_json
        self.labels = labels
        if from_json:
            self.load()
        else:
            self.amt_labels = len(self.labels)
            self.dirs = [PHOTO_DIR + label + "/" for label in self.labels]
            self.model = Sequential()
            self.model.add(Conv2D(64, kernel_size=3, activation="relu",
                                  input_shape=(PHOTO_ROWS, PHOTO_COLS, 1)))
            self.model.add(Conv2D(32, kernel_size=3, activation="relu"))
            self.model.add(Flatten())
            self.model.add(Dense(16, activation="relu"))
            self.model.add(Dense(self.amt_labels, activation="softmax"))
            self.model.compile(optimizer="adam", loss="cattegorical_crossentropy",
                               metrics="accuracy")

    def fetch_data(self):
        xs = []
        ys = []
        for ix, direc in enumerate(self.dirs):
            y = np.array([0 for i in range(self.amt_labels)])
            y[ix] = 1
            photos = [direc + fname for fname in os.listdir(direc)]
            for p in photos:
                photo = Image.open(p)
                resized = im.resize(PHOTO_ROWS, PHOTO_COLS)
                xs.append(np.array(resized))
                ys.append(y)
        return (xs, ys)

    def get_train_test(self, xs, ys, split=0.7):
        sh_xs, sh_ys = shuffle(xs, ys)
        len_xs = len(xs)
        train_size = int(len_xs * split)
        x_train, y_train = sh_xs[:train_size], sh_ys[:train_size]
        x_test, y_test = sh_xs[train_size + 1:len_xs], sh_ys[train_size + 1:len_xs]
        return (x_train, x_test, y_train, y_test)

    def train(self):
        xs, ys = self.fetch_data()
        x_train, x_test, y_train, y_test = self.get_train_test(xs, ys)
        self.model.fit(x_train, y_train, validation_data=(x_test, y_test),
                       epochs=3)

    def predict(self, x):
        prediction = self.model.predict(x)
        ix = np.argmax(prediction)
        return self.labels[ix]

    def save(self):
        with open(MODEL_DIR + "model.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights("model.h5")

    def load(self):
        with open(MODEL_DIR + "model.json", "w") as json_file:
            self.model = model_from_json(json_file.read())
        self.model.load_weights(MODEL_DIR + "model.h5")
