# system tools
import os
import sys
sys.path.append(os.path.join("../../.."))

# For visualisation
import pandas as pd
import numpy as np
import gensim.downloader

# Import classifier utility functions 
import utils.classifier_utils as clf

# Tools for machine learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# tensorflow-tools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D)
from tensorflow.keras.optimizers import SGD, Adam #Adam = optimerings algoritme
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2

# matplotlib
import matplotlib.pyplot as plt

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
    
def create_embedding_matrix(filepath, word_index, embedding_dim):
        
    vocab_size = len(word_index) + 1  # Adding again 1 because of the reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

                return embedding_matrix

# Creating the main function
def main(): 
            
    #Defining filepath
    filepath = '../Data_assignment6/Game_of_Thrones_Script.csv'

    #Reading csv file as data frame.
    dataframe = pd.read_csv(filepath)

    # Take sentence, season and names data and create np arrays.
    sentences = dataframe["Sentence"].values
    seasons = dataframe["Season"].values
    names = dataframe["Name"].values

    #Create the test and training data
    X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                        seasons,
                                                        test_size=0.25,
                                                        random_state=42)


    vectorizer = CountVectorizer()

    # We start with our training data.
    X_train_feats = vectorizer.fit_transform(X_train)
    # Then we run it on the test data.
    X_test_feats = vectorizer.transform(X_test)
    # I the create a list of the featured names.
    feature_names = vectorizer.get_feature_names()

    # Fitting the model.
    classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)

    y_pred = classifier.predict(X_test_feats)

    classifier_metrics = metrics.classification_report(y_test, y_pred)
    print(classifier_metrics)


    # Plotting and saving
    clf.plot_cm(y_test, y_pred, normalized=True)
    plt.savefig("../Output/actual_and_predicted.png")

    # Vectorize full dataset
    X_vect = vectorizer.fit_transform(sentences)

    # Initialise cross-validation method
    title = "Learning Curves (Logistic Regression)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # Run on data
    model = LogisticRegression(random_state=42)
    clf.plot_learning_curve(model, title, X_vect, seasons, cv=cv, n_jobs=4)
    plt.savefig("../Output/three_plots.png")

    
    # Define behaviour when called from command line
if __name__ =='__main__':
    main()