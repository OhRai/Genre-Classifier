import tensorflow as tf
from keras import models
from lstm_classifier import split_sets

def eval_model(model_tuple, X_test, y_test):
    model, name = model_tuple
    loss, acc = model.evaluate(X_test, y_test)
    print(f'The {name} model has an accuracy of {acc * 100:.2f}% and a loss of {loss:.2f}')

if __name__ == "__main__":
    # get test set
    _, _, X_test, _, _, y_test = split_sets(0.25, 0.2)

    # load models
    basic_model = (models.load_model('../genre classifier/models/all_samples/basic.h5'), 'Basic')
    cnn_model = (models.load_model('../genre classifier/models/all_samples/cnn.h5'), ' CNN')
    lstm_model = (models.load_model('../genre classifier/models/all_samples/lstm.h5'), 'LSTM')

    # evaluate models on test set
    eval_model(basic_model, X_test, y_test)
    eval_model(cnn_model, X_test, y_test)
    eval_model(lstm_model, X_test, y_test)
