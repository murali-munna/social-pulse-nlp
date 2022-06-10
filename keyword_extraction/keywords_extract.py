import os
import pandas as pd
import numpy as np
import re
import yake
import spacy
from keybert import KeyBERT
nlp = spacy.load('en_core_web_lg')
import pickle

brand_dict = {960: 'McDonalds', 911: 'Disney', 730: 'Robinhood', 646: 'Samsung', 590:'Netflix', 
              528: 'Ferrari', 385: 'WhatsApp', 273: 'Iphone', 931:'GooglePixel', 
              919:'Dell', 490:'Tesla', 109:'Facebook', 104:'Microsoft'}


def load_sentiment_emotion_data(path):
    """
    It takes a path to a directory of csv files, reads them in, and drops and renames some columns
    
    Args:
      path: the path to the folder containing the csv files for each brand
    
    Returns:
      An appended dataframe with the preprocessed columns
    """
    
    dfs = []
    for f in os.listdir(path):
        d = pd.read_csv(os.path.join(path, f))
        d['brand'] = brand_dict[int(f.split('_')[0])]
        dfs.append(d)

    df = pd.concat(dfs)

    df = df.drop(['Unnamed: 0',	'Unnamed: 0.1',	'Unnamed: 0.1.1', 'Keyword Search'], axis=1)

    df = df.rename({'topic': 'search_term', 
                    'time of creation': 'time_of_creation',
                    'sentiment prediction': 'sentiment_prediction',
                    'sentiment probability': 'sentiment_probability',
                    'emotion_prediction': 'emotion_prediction'}, axis=1)
    return df
    
def get_hashtags(text):
    """
    Extracts hashtags from text after performing text preprocessing like lowercasing, punctuation removal etc.
    
    Args:
      text: the text you want to extract hashtags from
    
    Returns:
      A list of hashtags
    """
    temp = text.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    hts = [part[1:] for part in temp.split() if part.startswith('#')]
    hts = [re.sub('[()!?]', ' ', x) for x in hts]
    hts = [re.sub('\[.*?\]',' ', x) for x in hts]
    hts = [re.sub("[^a-z0-9]"," ", x) for x in hts]
    hts = [x.strip() for x in hts]
    return list(set(hts))


def apply_hashtags(df):
    """
    It takes a dataframe as input, and returns a dataframe with a new column called 'hashtags' that
    contains a list of hashtags for each tweet
    
    Args:
      df: the dataframe
    
    Returns:
      A dataframe with a new column called 'hashtags'
    """
    df['hashtags'] = df['text'].apply(get_hashtags)
    return df


def text_clean_yake(text):
    """
    Cleans the text by removing mentions, hashtags, weblinks, websites, punctuation, non alpha numeric, and additional
    whitespaces
    
    Args:
      text: The text to be processed.
    
    Returns:
      A cleaned text
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
    temp = re.sub(' +', ' ', temp) # stripping additional whitespaces

    return temp


def get_ht_dict(df):
    """
    For each brand, get the hashtags from the twitter and reddit posts
    
    Args:
      df: the dataframe containing the data
    
    Returns:
      A dictionary of hashtags for each brand and stream
    """
    
    kw_ht = {}
    for brand in list(df['brand'].unique()):
        s = {}
        for stream in ['Twitter', 'Reddit']:
            # print(brand, stream)
            ht_lists = df[(df['brand']==brand) & (df['stream']==stream)]['text'].apply(get_hashtags).tolist()
            hts = [x for li in ht_lists if li for x in li if x]
            # print(hts)
            if hts:
                s[stream] = hts
        kw_ht[brand] = s
    
    return kw_ht


def get_kw_yake_dict(df):
    """
    For each brand, get the top 20 keywords from the YAKE keyword extractor for both Twitter and Reddit
    
    Args:
      df: dataframe with the text data
    
    Returns:
      A dictionary of YAKE keywords for each brand and stream
    """
    
    kw_yake = {}
    for brand in list(df['brand'].unique()):
        s = {}
        for stream in ['Twitter', 'Reddit']:
            # print(brand, stream)
            texts = df[(df['brand']==brand) & (df['stream']==stream)]['text_clean_yake'].tolist()
            texts = ' '.join([x.strip()+'.' for x in texts])
            kw_extractor = yake.KeywordExtractor(top=20, stopwords=None)
            keywords = kw_extractor.extract_keywords(texts)
            if keywords:
                s[stream] = list(list(zip(*keywords))[0])[::-1]
        kw_yake[brand] = s
    
    return kw_yake


def spacy_noun_chunks(doc_text, model = 'en_core_web_lg'):
    """
    Extracts SpaCy noun chunks from a given preprocessed document text.
    
    Let's try it out on the first document in the corpus
    
    Args:
      doc_text: the text you want to extract noun chunks from
      model: the model to use for the NLP. Defaults to en_core_web_lg
    
    Returns:
      A list of noun chunks
    """
    nlp = spacy.load(model)
    doc = nlp(doc_text)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    noun_chunks = list(set(noun_chunks))
    # noun_chunks = [chunk for chunk in noun_chunks if len(chunk.split())>1]
    rc = re.compile("“|”")
    noun_chunks = [rc.sub('', chunk) for chunk in noun_chunks]
    return noun_chunks

def get_noun_chunks_dict(df):
    """
    Extract the SpaCy noun chunks from the cleaned text posts
    
    Args:
      df: the dataframe containing the cleaned text data
    
    Returns:
      A dictionary of SpaCy noun chunks for each brand and stream
    """
    noun_chunks = {}
    for brand in list(df['brand'].unique()):
        s = {}
        for stream in ['Twitter', 'Reddit']:
            texts = df[(df['brand']==brand) & (df['stream']==stream)]['text_clean_yake'].tolist()
            texts = ' '.join([x.strip()+'.' for x in texts])
            text_noun_chunks = spacy_noun_chunks(texts)
            print(brand, stream, len(text_noun_chunks))
            s[stream] = text_noun_chunks
        noun_chunks[brand] = s
    
    return noun_chunks

def get_kw_kbnc_dict(df, kw_model, noun_chunks):
    """
    For each brand, extract the top 20 KeyBERT keywords from the text of each stream (Twitter and Reddit) using
    the noun chunks as candidates
    
    Args:
      df: the dataframe with the text data
      kw_model: the keyword extractor model
      noun_chunks: a dictionary of noun chunks for each brand and stream
    
    Returns:
      A dictionary of keyBERT keywords for each brand and stream
    """
    
    kw_kbnc = {}
    for brand in list(df['brand'].unique()):
        s = {}
        for stream in ['Twitter', 'Reddit']:
            texts = df[(df['brand']==brand) & (df['stream']==stream)]['text_clean_yake'].tolist()
            texts = ' '.join([x.strip()+'.' for x in texts])
            
            keywords = kw_model.extract_keywords(texts, 
                                            candidates = noun_chunks[brand][stream],
                                            # keyphrase_ngram_range=(1, 3), 
                                            stop_words=None, 
                                            top_n = 20,
                                            use_maxsum=False, nr_candidates=20,
                                            use_mmr=True, diversity=0.5)
            if keywords:
                kws = list(list(zip(*keywords))[0])

            # print(brand, stream, len(kws))
            if keywords:
                s[stream] = kws
    kw_kbnc[brand] = s
    
    return kw_kbnc


if __name__=='__main__':
    
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    
    df['text_clean_yake'] = df['text'].apply(text_clean_yake)
    
    kw_ht = get_ht_dict(df)
    
    kw_yake = get_kw_yake_dict(df)
    
    noun_chunks = get_noun_chunks_dict(df)
    
    kw_model = KeyBERT(model='all-MiniLM-L12-v2')
    
    kw_kbnc = get_kw_kbnc_dict(df, kw_model, noun_chunks)
    
    df['date_time'] = pd.to_datetime(df['time_of_creation'])
    df['date'] = pd.to_datetime(df['time_of_creation']).dt.date
    df['days_diff'] = (pd.Timestamp.now() - df['date_time']).dt.days
    df['time_period'] = np.where(df['days_diff']<=7, 1,
                        np.where(df['days_diff']<=31, 2,
                        np.where(df['days_diff']<=180, 3,
                        np.where(df['days_diff']<=365, 4,
                        np.where(df['days_diff']<=365*5, 5, 6)))))
    df['posts'] = 1
    df['votes'] = df['ups'].fillna(df['favorite_count'])
    df['sentiment'] = df['sentiment_prediction'].replace({1:'Positive', 0:'Neutral', -1:'Negative'})
    df['sentiment'] = pd.Categorical(df['sentiment'], ["Negative", "Neutral", "Positive"])
    df = df.rename({'emotion prediction': 'emotion'}, axis=1)
    df['emotion'] = pd.Categorical(df['emotion'], ['joy', 'sadness', 'surprise', 'anger', 'fear'])
    
    with open('streamlit_app/data/df.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('streamlit_app/data/kw_ht.pkl', 'wb') as handle:
        pickle.dump(kw_ht, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('streamlit_app/data/kw_yake.pkl', 'wb') as handle:
        pickle.dump(kw_yake, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('streamlit_app/data/noun_chunks.pkl', 'wb') as handle:
        pickle.dump(noun_chunks, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('streamlit_app/data/kw_kbnc.pkl', 'wb') as handle:
        pickle.dump(kw_kbnc, handle, protocol=pickle.HIGHEST_PROTOCOL)
