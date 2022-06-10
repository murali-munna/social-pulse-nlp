import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import re
from spacy.lang.en.stop_words import STOP_WORDS
import pickle


def getData(twitterFname, redditFname):
    """
    It reads in the two data files, combines them, and then splits them into training and test sets

    Args:
      twitterFname: the name of the file containing the twitter data
      redditFname: The name of the file containing the Reddit data.

    Returns:
      train_X, test_X, train_y, test_y
    """

    # https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?resource=download
    # dfT = pd.read_csv('archive/Twitter_Data.csv')
    # dfR = pd.read_csv('archive/Reddit_Data.csv')
    dfT = pd.read_csv(twitterFname)
    dfR=pd.read_csv(redditFname)
    dfR.rename(columns={"clean_comment":"clean_text"},inplace=True)
    df = pd.concat((dfT,dfR))
    df.dropna(axis=0,inplace=True)
    minCatCount = min(df.category.value_counts())
    balanced_df = pd.concat((df[df.category == -1][:minCatCount],df[df.category == 0][:minCatCount],df[df.category == 1][:minCatCount]))
    balanced_df.category.value_counts()

    comment, category = [],[]
    for row in df.iterrows():
        comment.append(row[1]['clean_text'])
        category.append(row[1]['category'])

    train_X, test_X, train_y, test_y = train_test_split([i for i in comment], [i for i in category], test_size=0.25, random_state=1)

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
    return vectorizer, train_X_vectors, test_X_vectors


def train_SVM(train_X_vectors, train_y, test_X_vectors, test_y):
    """
    It takes in the training and testing data, and returns a trained SVM model

    Args:
      train_X_vectors: The training data
      train_y: the labels for the training data
      test_X_vectors: the vectorized test data
      test_y: the actual labels of the test set

    Returns:
      the trained model.
    """
    clf_svm = LinearSVC()
    clf_svm.fit(train_X_vectors, train_y)
    clf_prediction = clf_svm.predict(test_X_vectors)

    print(f'Accuracy: {clf_svm.score(test_X_vectors, test_y)}')
    print(f'Accuracy: {f1_score(test_y, clf_prediction, average=None, labels=[-1, 0, 1])}')
    return clf_svm

def modelPredict(clf_svm, vectorizer):
    """
    > The function takes in a trained model and a vectorizer and returns the prediction of the model on the text 'How about
    this text?'

    Args:
      clf_svm: the trained model
      vectorizer: This is the vectorizer that we trained on our training data.

    Returns:
      The model is returning the predicted class of the input text.
    """
    return clf_svm.predict(vectorizer.transform(['How about this text?']))

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
    temp = re.sub("'", "", temp)  # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+", "", temp)  # mentions
    temp = re.sub("#[A-Za-z0-9_]+", "", temp)  # hashtags
    temp = re.sub(r"http\S+", "", temp)  # weblinks
    temp = re.sub(r"www.\S+", "", temp)  # websites
    temp = re.sub('[()!?]', ' ', temp)  # punctuation
    temp = re.sub('\[.*?\]', ' ', temp)
    temp = re.sub("[^a-z0-9]", " ", temp)  # non alpha numeric
    temp = temp.split()
    temp = [w for w in temp if not w in STOP_WORDS]
    temp = " ".join(word for word in temp)
    return temp

def appendData():
    """
    It takes the data from the three datasets, cleans the text, and then uses the sentiment model to predict the sentiment
    of each tweet

    Returns:
      the sentiment prediction and the sentiment probability for each row in the dataframe.
    """
    filenameModel = './sentimentModel.sav'
    filenameVectorizer = './vectorizer.sav'
    sentimentModel = pickle.load(open(filenameModel, 'rb'))
    Vectorizer = pickle.load(open(filenameVectorizer, 'rb'))

    dNumbers = [104, 931, 109]

    for dataNumber in dNumbers:
        print("Appending dataset {}".format(dataNumber))
        data = pd.read_csv('./data/scrapeDF_{}.csv'.format(dataNumber))

        data["sentiment prediction"] = None
        data["sentiment probability"] = None

        for idx, row in data.iterrows():
            cleanedText = clean_text(row["text"])
            prediction = sentimentModel.predict(Vectorizer.transform([cleanedText]))
            probabilities = sentimentModel.predict_proba(Vectorizer.transform([cleanedText]))
            data.at[idx, "sentiment prediction"] = prediction[0]
            data.at[idx, "sentiment probability"] = probabilities[0][int(prediction[0]) + 1]
            if idx % int(len(data.index) / 10) == 0 and idx > 0:
                print('{} / {} complete'.format(idx, len(data.index)))

        data.to_csv('./appended/{}_sentimentAppend.csv'.format(dataNumber))
    return