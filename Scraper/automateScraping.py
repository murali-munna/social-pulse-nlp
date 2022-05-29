import numpy as np
import praw
import datetime
import random
import os
import tweepy
import pandas as pd


#Dataframe parameters
RedditCount = 1000
TwitterCount = 1000
columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']
locationDict = {'San Diego': '32.71642383476381,-117.16143519352777,20km',
                    'San Jose': '37.42144707383498,-121.90427070796471,20km',
                    'New York': '40.72604974544931,-74.00030492668076,20km',
                    'Seattle': '47.610165758968876,-122.32559045176778,20km',
                    'Chicago': '41.90674120873151,-87.63899671296997,20km'}
searchType = ['hot','top','new','controversial','relevance']
timeframe = 'year'

#Twitter
BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
twitter_auth = tweepy.OAuth2BearerHandler(BEARER_TOKEN)
twitter_api = tweepy.API(twitter_auth)

#Reddit
client_id = os.environ.get("client_id")
client_secret = os.environ.get("client_secret")
user_agent = os.environ.get("user_agent")
reddit_read_only = praw.Reddit(client_id=client_id,client_secret=client_secret,user_agent=user_agent)


def TwitterRoutine(Redd_Twitt_queries, sname):
    '''

    :param Redd_Twitt_queries:
    :return:
    '''
    df = pd.DataFrame(columns=columns)
    Twitt_qLen = len(Redd_Twitt_queries[1])
    Twitt_count = int(TwitterCount / len(locationDict.keys()) / Twitt_qLen)

    pastLoc = None
    i = 0
    for location in locationDict.items():
        locName = location[0]
        if locName != pastLoc:
            print('Searching Twitter for: ' + sname + ' in location: ' + locName)
        pastLoc = locName
        for phrase in Redd_Twitt_queries[1]:
            i += 1
            geo = location[1]
            data = []

            for tweet in tweepy.Cursor(twitter_api.search_tweets, phrase, geocode=geo, count=100).items(max(Twitt_count, Twitt_count+(i*Twitt_count-df.index.size))):
                dt_tweet = tweet.created_at.isoformat()
                data.append([phrase, 'Twitter', tweet.text, dt_tweet[:-6], geo, None, tweet.favorite_count])
            df = pd.concat((df,pd.DataFrame(data, columns=columns)))
        print('Twitter Routine>>> df size: ' + str(df.index.size))

    return df


def RedditRoutine(Redd_Twitt_queries, sname):
    '''

    :param Redd:
    :return:
    '''
    df = pd.DataFrame(columns=columns)
    Redd_qLen = len(Redd_Twitt_queries[0])
    Reddit_count = int(RedditCount / len(searchType) / Redd_qLen)


    for search in searchType:
        print('Searching Reddit in ' + sname + ' with search type: ' + search)
        for phrase in Redd_Twitt_queries[0]:
            data = []
            subreddit = reddit_read_only.subreddit(phrase)
            api_return = subreddit.search(sname, time_filter=timeframe, sort=search)
            submissions = list(api_return)
            for submission in submissions:
                comments = submission.comments
                for comment in comments[0:Reddit_count - 1]:
                    try:
                        createdAt = datetime.datetime.fromtimestamp(comment.created_utc)
                        topic = '{} {}'.format(sname, phrase) if phrase is not None else '{}'.format(
                            sname)
                        data.append([topic, 'Reddit', comment.body, createdAt.isoformat(), None, comment.ups, None])
                        if len(data)%50==0:
                            print('Collected ' + str(len(data)) + ' data points...')
                    except:
                        continue

            try:
                data = list(map(data.__getitem__, [random.randrange(0, len(data) - 1) for i in range(min(len(data), Reddit_count))]))
                print('Adding ' + str(min(len(data), Reddit_count)) + ' to df with phrase: ' + phrase)
            except:
                print('No return...')

            df = pd.concat((df, pd.DataFrame(data, columns=columns)))
        print('RedditRoutine>>> df size: ' + str(df.index.size))

    return df


def autoScrape(industryDict):
    '''

    :return:
    '''
    df = pd.DataFrame(columns=columns)
    Redd_Twitt_queries = list(industryDict.items())[0][1]
    key = list(industryDict.items())[0][0]

    dfAdd = TwitterRoutine(Redd_Twitt_queries, key)
    df = pd.concat((df, dfAdd))
    print('autoScrape>>> df size: ' + str(df.index.size))


    dfAdd = RedditRoutine(Redd_Twitt_queries, key)
    df = pd.concat((df, dfAdd))
    print('autoScrape>>> df size: ' + str(df.index.size))

    filename = 'scrapeDF_' + str(np.random.randint(100, 999)) + '.csv'
    df = df.reset_index(drop=True)
    df.to_csv(filename)
    print('Exported filename: ' + filename)

    return
