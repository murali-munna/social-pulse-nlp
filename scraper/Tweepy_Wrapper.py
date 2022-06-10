import os
import tweepy
import pandas as pd

BEARER_TOKEN = os.environ.get("BEARER_TOKEN")

twitter_auth = tweepy.OAuth2BearerHandler(BEARER_TOKEN)
twitter_api = tweepy.API(twitter_auth)


def tweetScrape(count=50, geocode = '32.71642383476381,-117.16143519352777,20km'):
    """
    This function takes in a search term and a geocode, and returns a dataframe of tweets that match the search term within
    the geocode

    Args:
      count: The number of tweets to return per page, up to a maximum of 100. Defaults to 15. This was formerly the “rpp”
    parameter in the old Search API. Defaults to 50
      geocode: The latitude and longitude of the center of the search area, followed by the radius of the search area.
    Defaults to 32.71642383476381,-117.16143519352777,20km

    Returns:
      A dataframe with the columns: topic, stream, text, time of creation, location, ups, favorite_count
    """

    columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']
    data = []

    # geocode = input('Please input geocode (example: 37.7821120598956,-122.400612831116,3km)')
    query = input('Please input search term (hashtag search, include \'#\'):')
    print('Scraping Twitter...')
    api_response = twitter_api.search_tweets(q=query, geocode=geocode, count=count)

    for tweet in api_response:
        dt_tweet = tweet.created_at.isoformat()
        data.append([query, 'Twitter',tweet.text,dt_tweet[:-6],geocode,None,tweet.favorite_count])

    return pd.DataFrame(data,columns=columns)

def userScrape(count=50):
    '''

    :return:
    '''

    raise NotImplementedError



