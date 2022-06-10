import os
import praw
import pandas as pd
import datetime
import random

def subredditScrape(count=50, commentParentMax=3):
    """
    This function takes in a count of comments to return, a commentParentMax (the number of parent comments to search
    through), a phraseSearch (whether or not to search for a phrase), a phrase (the phrase to search for), a hottopnewcont
    (the type of stream to search through), a timeframe (the timeframe to search through), and a subreddit_name (the
    subreddit to search through)

    Args:
      count: the number of comments to return. Defaults to 50
      commentParentMax: The number of submissions to scrape from. Defaults to 3

    Returns:
      A dataframe with the following columns:
        - Topic
        - Source
        - Text
        - Date
        - Author
        - Upvotes
        - Link
    """



def redditorScrape(count=50, commentParentMax=5):
    """
    This function takes in a count of how many comments you want to scrape, and a commentParentMax of how many parent
    comments you want to scrape from. It then asks you if you want to do a phrase search, and if so, what the phrase is. It
    then asks you what type of stream you want to scrape from (hot, top, new, controversial, relevance), and if you chose
    top or controversial, it asks you what time frame you want to scrape from. It then asks you what redditor you want to
    scrape from, and if you want to keep the redditor it found. It then scrapes the comments from the redditor, and returns
    a pandas dataframe of the comments

    Args:
      count: the number of comments you want to scrape. Defaults to 50
      commentParentMax: The maximum number of parent comments to scrape. Defaults to 5

    Returns:
      A dataframe with the following columns:
        - body: the text of the comment
        - createdAt: the date and time the comment was created
        - subreddit: the subreddit the comment was posted in
        - ups: the number of upvotes the comment has
        - url: the url of the comment
    """
