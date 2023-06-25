# Genre Classifier
This project was created to test out different models and train them to classify different music genres. The models that were used are: a simple model, a CNN model, and an LSTM model. 

Each model is trained for 100 epochs, with a batch size of 32, and a learning rate of 0.0001 to evaluate their performance on the task. However note that each model will not perform well with this specific hyperparameter, as it may cause overfitting. Thus, it is important to test around with hyperparameters that will give you the best outcome. 

## Dataset

The classifier is trained on the GTZAN dataset, which consists of audio files from various music genres. It contains 1000 audio files, each being 30s long, of 10 different genres. The dataset can be found here: [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

## Preprocessing

The `preprocess.py` script is responsible for preprocessing the audio data. It loads the audio files from the dataset, extracts Mel Frequency Cepstral Coefficients (MFCCs) as features, and stores them in an array along with their corresponding labels in a JSON file.

## Basic Model

The `basic_classifier.py` script implements the most basic and simple model architecture. It is comprised of a couple of dense layers with some dropout to lower the cause of overfitting the model. It also uses some very powerful activations such as softmax and ReLU. This was however the worst performing model, having an accuracy of around 60%. 

Looking at the accuracy and loss graph for this model, it seems to show signs of overfitting at around 60~ epochs.

## CNN Model

The `cnn_classifier.py` script implements the CNN model architecture. Because the audio files are converted to MFCC spectrograms, they are in the form of an image, and CNN models are great for training using an image dataset. This model included 2D convolutional layers, max pooling layers, and batch normalization. This model did much better than the basic model, having an accuracy of about 85%.

Looking at the accuracy and loss graph for this model, it seems to show signs of overfitting very early on during the training. By creating a simpler model or changing the hyperparameters, it can fix this problem.

## LSTM Model

The `lstm_classifier.py` script implements the LSTM model architecture. RNNs are also another type of model that is commonly used for audio datasets, and this performed nearly as well as the CNN model. It had accuracy of around 80%. 

Looking at the accuracy and loss graph for this model, it seems to show signs of overfitting similar to the CNN model. Again this can be fixed by simplifying the model or changing its hyperparameters. 

# Future Plans
I intend on creating a program to detect which genre a song is for the game osu!. I plan on scraping  many songs from there with the genre attached to it by calling their api (which can be found on https://osu.ppy.sh/docs/index.html) This way if a map designer does not know what genre a certain song is, this program can predict it. 

# Usage and Testing

Before running the genre classifier, make sure you have the required dependencies installed. You can install them by following these steps:

Clone the repository:

  ```shell
   git clone https://github.com/your-username/genre-classifier.git
   cd genre-classifier
```
(Optional) Create a virtual environment:
```shell
python -m venv env
source env/bin/activate`
```
Install the required dependencies: 
```shell
pip install -r requirements.txt
```

Now you can test around in each python script. 

## WARNING
When running the preprocess.py script, it will create a very large json file. The one created for the full 1000 audio file dataset was around 600 mb. Also note that in the jazz genre, the jazz0054.wav file is corrupted, so please remove this. 