from pandas import DataFrame
from scraper.Praw_Wrapper import subredditScrape
import pytest

def test_subredditScrape()->DataFrame:
    newDF = subredditScrape(50, 3)
    assert isinstance(newDF, DataFrame)

def test_subredditScrape()->DataFrame:
    newDF = subredditScrape(50, 3)
    assert isinstance(newDF, DataFrame)
