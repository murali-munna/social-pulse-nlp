from pandas import DataFrame
from scraper.automateScraping import TwitterRoutine, RedditRoutine
import pandas as pd
import pytest

# test TwitterRoutine
def test_TwitterRoutine()->DataFrame:
    industryDict = {'Apple': [['iphone 14', 'Apple Phones'], ['#iphone14', 'iphone 14', 'Apple Phones', 'iphone']]}
    columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']
    df = pd.DataFrame(columns=columns)
    Redd_Twitt_queries = list(industryDict.items())[0][1]
    key = list(industryDict.items())[0][0]
    newDF = TwitterRoutine(Redd_Twitt_queries, key)
    assert isinstance(newDF, DataFrame)

# test RedditRoutine
def test_RedditRoutine()->DataFrame:
    industryDict = {'Apple': [['iphone 14', 'Apple Phones'], ['#iphone14', 'iphone 14', 'Apple Phones', 'iphone']]}
    columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']
    df = pd.DataFrame(columns=columns)
    Redd_Twitt_queries = list(industryDict.items())[0][1]
    key = list(industryDict.items())[0][0]
    newDF = RedditRoutine(Redd_Twitt_queries, key)
    assert isinstance(newDF, DataFrame)

# test autoScrape
def test_autoScrape()->None:
    industryDict = {'Apple': [['iphone 14', 'Apple Phones'], ['#iphone14', 'iphone 14', 'Apple Phones', 'iphone']]}
    columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']
    df = pd.DataFrame(columns=columns)
    Redd_Twitt_queries = list(industryDict.items())[0][1]
    key = list(industryDict.items())[0][0]
    newRedditDF = RedditRoutine(Redd_Twitt_queries, key)
    newTwitterDF = TwitterRoutine(Redd_Twitt_queries, key)
    assert isinstance(newRedditDF, DataFrame) and isinstance(newTwitterDF, DataFrame)
