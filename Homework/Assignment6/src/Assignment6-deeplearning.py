# System
import os
import sys
sys.path.append(os.path.join("..", "..", ".."))

# Pandas, Numpy, Gensim
import pandas as pd
import numpy as np
import gensim.downloader

# Import classifier utility functions. This code was developed for use in class and has been adapted for this project”
import utils.classifier_utils as clf

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam #Adam = optimization algoritm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# Matplotlib
import matplotlib.pyplot as plt

# Function for plotting model history
    # H: Model history
    # Epochs: Number of epochs the model was trained on
def plot_history(H, epochs):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # A function to read in saved GloVe embeddings and create and embedding matrix
        # Filepath: path to GloVe embedding
        # Word_index: indices from keras Tokenizer
        # Embedding_dim: dimensions of keras embedding layer
    def create_embedding_matrix(filepath, word_index, embedding_dim):
        
   
        vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix
    
    def main():
    
    
        #Defining filepath
        filepath = '../Data_assignment6/Game_of_Thrones_Script.csv'

        #Reading csv file as data frame.
        dataframe = pd.read_csv(filepath)

        # Take sentence, season and names data and create np arrays.
        sentences = dataframe["Sentence"].values
        seasons = dataframe["Season"].values
        names = dataframe["Name"].values

        #Create test and training data
        X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                            seasons,
                                                            test_size=0.25,
                                                            random_state=42)


        vectorizer = CountVectorizer()

        # We start with our training data.
        X_train_feats = vectorizer.fit_transform(X_train)

        # Then we run it on the test data.
        X_test_feats = vectorizer.transform(X_test)

        # I then create a list of the featured names. 
        feature_names = vectorizer.get_feature_names()



        y_train = pd.factorize(y_train)[0]
        y_test = pd.factorize(y_test)[0]

        l2 = L2(0.0001)

        # Initialize tokenizer
        tokenizer = Tokenizer(num_words=5000) #vocabulary on 5000 words
        # Fit to training data
        tokenizer.fit_on_texts(X_train)

        # Tokenized training and test data
        X_train_toks = tokenizer.texts_to_sequences(X_train) # Convert to sequences
        X_test_toks = tokenizer.texts_to_sequences(X_test)

        # Vocabulary size
        vocab_size = len(tokenizer.word_index) + 1

        # Inspect
        print(X_train[2])
        print(X_train_toks[2])

        # Define embedding size
        embedding_dim = 50

        embedding_matrix = create_embedding_matrix('../Data_assignment6/glove.6B.50d.txt',
                                                   tokenizer.word_index, 
                                                   embedding_dim)
        # Max length for a doc
        maxlen = 100

        # Pad training data to maxlen
        X_train_pad = pad_sequences(X_train_toks, 
                                    padding='post', # We choose "post" instead of "pre"
                                    maxlen=maxlen)
        # Pad testing data to maxlen
        X_test_pad = pad_sequences(X_test_toks, 
                                   padding='post', 
                                   maxlen=maxlen)

        # New model
        model = Sequential()

        # Embedding -> CONV+ReLU -> MaxPool -> FC+ReLU -> Out
        model.add(Embedding(vocab_size,                  # vocab size from Tokenizer()
                            embedding_dim,               # embedding input layer size
                            weights=[embedding_matrix],  # pretrained embeddings
                            input_length=maxlen,         # maxlen of padded doc
                            trainable=True))             # trainable embeddings
        model.add(Conv1D(128, 5, 
                        activation='relu',
                        kernel_regularizer=l2))          # Using L2 regularization 
        model.add(GlobalMaxPool1D())
        model.add(Dense(10, activation='relu', kernel_regularizer=l2))
        model.add(Dense(1, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', # Categorical_crossentropy because we don't have a binary.
                      optimizer="adam", # Using the adam-optimizer
                      metrics=['accuracy'])

        # Print the summary
        model.summary()


        history = model.fit(X_train_pad, y_train,
                            epochs=20,
                            verbose=False,
                            validation_data=(X_test_pad, y_test),
                            batch_size=10)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_train_pad, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

        # Plot the model
        plot_history(history, epochs = 20)
        plt.savefig("../Output/training_loss5.png")
    
    # Define behaviour when called from command line
    if __name__ =='__main__':
        
        main()