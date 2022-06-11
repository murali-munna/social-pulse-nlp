from pandas import DataFrame
from scraper.Tweepy_Wrapper import tweetScrape
import pytest

def test_tweetScrape()->DataFrame:
    newDF = tweetScrape(50, '32.71642383476381,-117.16143519352777,20km')
    assert isinstance(newDF, DataFrame)
