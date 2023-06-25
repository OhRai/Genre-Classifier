import json
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

DATASET_PATH = 'data.json'
EPOCHS = 100
BATCH_SIZE = 32

def load_data(dataset_path):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists to np arrays
    inputs = np.array(data['mfcc'])
    targets = np.array(data['labels'])

    return inputs, targets

def split_sets(test_size, val_size):
    X, y = load_data(DATASET_PATH)

    # create split (train/test) (train/val)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=val_size)

    # make X into 4d array -> (num_samples, 130, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape):
    model = Sequential()
    
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

def predict(model, X, y):
    # X needs to be 4D -> (1, 130, 13, 1)
    X = X[np.newaxis, ...]
    
    prediction = model.predict(X) 
    prediction_index = np.argmax(prediction, axis=1)
    print(f'Expected Index: {y}, Predicted Index: {prediction_index}')

def plot_hist(hist):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(hist.history['accuracy'], label='train accuracy')
    axs[0].plot(hist.history['val_accuracy'], label='test accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Accuracy eval')

    # error subplot
    axs[1].plot(hist.history['loss'], label='train loss')
    axs[1].plot(hist.history['val_loss'], label='test loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Loss eval')

    fig.suptitle('CNN Model')

    plt.show()
      
if __name__ == '__main__':
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split data into train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_sets(0.25, 0.2)
    
    # build the model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compile model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy')
    model.summary()

    # train the model
    hist = model.fit(X_train, y_train, 
                     validation_data=(X_validation, y_validation),
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZE)
    model.save('cnn.h5')

    # evalute
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Accuracy on test set is: {test_accuracy}')
        
    # plot history
    plot_hist(hist)