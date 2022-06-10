from pandas import DataFrame
from keyword_extraction.keywords_extract import *

def test_load_sentiment_emotion_data()->DataFrame:
    path = 'data/Emotion Appended'
    newDF = load_sentiment_emotion_data(path)
    assert isinstance(newDF, DataFrame)

def test_get_hashtags()->list:
    text = "#hashtag is a test"
    res = get_hashtags(text)
    assert isinstance(res, list)
    assert res == ['hashtag']

def test_apply_hashtags()->DataFrame:
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    assert isinstance(df, DataFrame)

def test_text_clean_yake(text)->str:
    text = "#hashtag is a test"
    res = text_clean_yake(text)
    assert isinstance(res, str)
    assert res == ' is a test'

def test_get_ht_dict()->dict:
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    kw_ht = get_ht_dict(df)
    assert isinstance(kw_ht, dict)

def test_get_kw_yake_dict()->dict:
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    kw_yake = get_kw_yake_dict(df)
    assert isinstance(kw_yake, dict)

def test_spacy_noun_chunks()->list:
    text = "#hashtag is a test"
    res = spacy_noun_chunks(text, 'en_core_web_lg')
    assert isinstance(res, list)

def test_get_noun_chunks_dict()->dict:
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    res = get_noun_chunks_dict(df)
    assert isinstance(res, dict)

def test_get_kw_kbnc_dict()->dict:
    path = 'data/Emotion Appended'
    df = load_sentiment_emotion_data(path)
    df = apply_hashtags(df)
    df['text_clean_yake'] = df['text'].apply(text_clean_yake)
    kw_ht = get_ht_dict(df)
    kw_yake = get_kw_yake_dict(df)
    noun_chunks = get_noun_chunks_dict(df)
    kw_model = KeyBERT(model='all-MiniLM-L12-v2')
    res = get_kw_kbnc_dict(df, kw_model, noun_chunks)
    assert isinstance(res, dict)