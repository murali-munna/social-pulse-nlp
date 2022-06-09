import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import os
import nltk
import itertools
import collections
import networkx as nx

def generate_train_test(file_name):
    """
    It takes a file name as input, reads the file, and returns the train and test data
    
    Args:
      file_name: The name of the file you want to read in.
    
    Returns:
      train_X, test_X, train_y, test_y
    """
    df = pd.read_csv(file_name)
    comment, category = [],[]
    for row in df.iterrows():
        comment.append(row[1]['text'])
        category.append(row[1]['label'])
    train_X, test_X, train_y, test_y = train_test_split([i for i in comment], [i for i in category], test_size=0.2, random_state=1)
    return train_X, test_X, train_y, test_y

def train_vectorizer(train_X, test_X):
    """
    It takes in a list of strings (train_X) and a list of strings (test_X) and returns a list of vectors
    (train_X_vectors) and a list of vectors (test_X_vectors)
    
    Args:
      train_X: the training data
      test_X: The test data
    
    Returns:
      The vectorizer is being returned.
    """
    vectorizer = TfidfVectorizer()
    train_X_vectors = vectorizer.fit_transform(train_X)
    test_X_vectors = vectorizer.transform(test_X)
    return train_X_vectors, test_X_vectors

def train_test(train_data):
    """
    It takes in a dataframe, splits it into train and test sets, vectorizes the text, trains a linear
    SVM, and returns the trained model
    
    Args:
      train_data: the dataframe that contains the data to be trained on
    
    Returns:
      The trained model
    """
    train_X, test_X, train_y, test_y = generate_train_test(train_data)
    train_X_vectors, test_X_vectors = train_vectorizer(train_X, test_X)
    clf_svm = LinearSVC(verbose=2, random_state=1)
    clf_svm.fit(train_X_vectors, train_y)
    clf_prediction = clf_svm.predict(test_X_vectors)

    print(f'Accuracy: {clf_svm.score(test_X_vectors, test_y)}')
    print(f'Accuracy: {f1_score(test_y, clf_prediction, average=None)}')
    return clf_svm

def single_predict(clf_svm, test_str):
    """
    It takes a trained model and a string as input, and returns a prediction
    
    Args:
      clf_svm: the trained model
      test_str: a string of text to be classified
    
    Returns:
      The prediction of the class of the test string.
    """
    pred = clf_svm.predict(vectorizer.transform(test_str))
    return pred

def clean_text(text):
    """
    It takes a string as input, and returns a string with all the non-alphanumeric characters removed,
    and all the words converted to lowercase
    
    Args:
      text: The text to be cleaned
    
    Returns:
      A string
    """
    temp = text.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp) # mentions
    temp = re.sub("#[A-Za-z0-9_]+","", temp) # hashtags
    temp = re.sub(r"http\S+", "", temp) # weblinks
    temp = re.sub(r"www.\S+", "", temp) # websites
    temp = re.sub('[()!?]', ' ', temp) # punctuation
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp) # non alpha numeric
    temp = temp.split()
    temp = [w for w in temp if not w in STOP_WORDS]
    temp = " ".join(word for word in temp)
    return temp

def csv_update(csv_list, emotion_cat, train_data):
    """
    This function takes in a list of csv files, a list of emotion categories, and a training data set.
    It then uses the training data to train a classifier, and then uses the classifier to predict the
    emotion category of each row in the csv files. It then appends the emotion category to the csv files
    
    Args:
      csv_list: a list of csv files to be updated
      emotion_cat: a list of emotions
      train_data: the training data
    
    Returns:
      Nothing is being returned.
    """
    clf_svm = train_test(train_data)
    for name in csv_list:  
        data_temp = pd.read_csv(name)
        data_temp['text_clean'] = data_temp['text'].apply(clean_text)
        comment = []
        for row in data_temp.iterrows():
            comment.append(row[1]['text_clean'])
        comment = np.asarray(comment)
        clf_prediction = single_predict(clf_svm, comment)
        emotion_col = [emotion_cat[idx] for idx in clf_prediction]
        data_temp['emotion prediction'] = emotion_col
        # hts = data_temp['text'].apply(get_hashtags).tolist()
        # hts = [j for i in hts for j in i]
        # data_temp['hashtags'] = hts
        data_temp.to_csv(name[:4] + 'emotionAppend.csv')
    return 

if __name__ == "__main__":
    train_data = "training.csv"
    csv_list = ['104_sentimentAppend.csv', '109_sentimentAppend.csv', '490_sentimentAppend.csv', '730_sentimentAppend.csv', '919_sentimentAppend.csv', '931_sentimentAppend.csv']
    emotion_cat = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    csv_update(csv_list, emotion_cat, train_data)