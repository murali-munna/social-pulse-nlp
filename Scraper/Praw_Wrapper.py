import os
import praw
import pandas as pd
import datetime
import random

columns=['topic','stream','text', 'time of creation', 'location', 'ups', 'favorite_count']

client_id = os.environ.get("client_id")
client_secret = os.environ.get("client_secret")
user_agent = os.environ.get("user_agent")
reddit_read_only = praw.Reddit(client_id=client_id,client_secret=client_secret,user_agent=user_agent)

def subredditScrape(count=50, commentParentMax=3):
    '''
    This function returns a pd.Dataframe as a result of scraping one or multiple subreddits
    https://praw.readthedocs.io/en/stable/code_overview/models/subreddit.html
    :param subreddits:
    :return:
    '''
    data = []

    phraseSearch = input('Is this a phrase search? (y/n): ')
    phrase = None
    if phraseSearch == 'y':
        phrase = input('Please input search phrase: ')

    relevanceTag = ', relevance' if phraseSearch == 'y' else ''
    hottopnewcont = input('Please specify stream type (hot, top, new, controversial{}): '.format(relevanceTag))
    timeframe = 'all'
    if hottopnewcont not in ('hot', 'new', 'relevance'):
        timeframe = input('Please input time filter (all,hour,day,week,month,year): ')



    subreddit_name = input('Please specify a subreddit or comma separated list of subreddits (\'all\' if reddit-wide): ')
    subreddit_name = subreddit_name.split(',')
    for i in range(len(subreddit_name)):
        subreddit_name[i] = subreddit_name[i].strip()

    for srName in subreddit_name:
        subreddit = reddit_read_only.subreddit(srName)
        toKeep = input('Subreddit ' + srName + ' , found: ' + subreddit.display_name + ' Keep? (y/n):')
        if toKeep != 'y':
            continue
        print('Scraping Reddit...')

        if phraseSearch == 'n':
            match hottopnewcont:
                case 'hot':
                    api_return = subreddit.hot(limit=commentParentMax)
                case 'top':
                    api_return = subreddit.top(time_filter=timeframe, limit=commentParentMax)
                case 'new':
                    api_return = subreddit.new(limit=commentParentMax)
                case 'controversial':
                    api_return = subreddit.controversial(time_filter=timeframe, limit=commentParentMax)
                case _:
                    print('Invalid stream type')
                    return None
        elif phraseSearch == 'y':
            api_return = subreddit.search(phrase, time_filter=timeframe, sort=hottopnewcont)

        else:
            print('Invalid input')
            return None

    submissions = list(api_return)
    for submission in submissions:
        comments = submission.comments
        for comment in comments[0:count-1]:
            try:
                createdAt = datetime.datetime.fromtimestamp(comment.created_utc)
                topic = '{} {}'.format(subreddit_name,phrase) if phrase is not None else '{}'.format(subreddit_name)
                data.append([topic, 'Reddit', comment.body, createdAt.isoformat(),None,comment.ups,None])
            except:
                continue

    # shuffle and return 'count' number of entries. Data is accumulated by sequential submissions but this
    # allows for random sampling across multiple submissions.
    print('Shuffling, returning {} number of entries...'.format(count))
    try:
        data = list(map(data.__getitem__, [random.randrange(0,len(data)-1) for i in range(count)]))
    except:
        print('No return...')

    return pd.DataFrame(data,columns=columns)



def redditorScrape(count=50, commentParentMax=5):
    '''
    This function returns a pd.Dataframe as a result  of scraping one or multiple redditors

    ...
    This currently does not work possibly because the reddit_read_only does not have authority to view specific user data
    ...

    https://praw.readthedocs.io/en/stable/code_overview/models/redditor.html
    :return:
    '''
    data = []

    phraseSearch = input('Is this a phrase search? (y/n): ')
    phrase = None
    if phraseSearch == 'y':
        phrase = input('Please input search phrase: ')

    relevanceTag = ', relevance' if phraseSearch == 'y' else ''
    hottopnewcont = input('Please specify stream type (hot, top, new, controversial{}): '.format(relevanceTag))
    timeframe = 'all'
    if hottopnewcont not in ('hot', 'new', 'relevance'):
        timeframe = input('Please input time filter (all,hour,day,week,month,year): ')

    redditor_name = input('Please input redditor name(s):')
    redditor_name = redditor_name.split(',')
    for i in range(len(redditor_name)):
        redditor_name[i] = redditor_name[i].strip()

    for rdName in redditor_name:
        redditor = reddit_read_only.redditor(rdName)
        toKeep = input('Redditor ' + rdName + ' , found: ' + redditor.name + ' Keep? (y/n):')
        if toKeep != 'y':
            continue

        if phraseSearch == 'n':
            match hottopnewcont:
                case 'hot':
                    api_return = redditor.hot(limit=commentParentMax)
                case 'top':
                    api_return = redditor.top(time_filter=timeframe, limit=commentParentMax)
                case 'new':
                    api_return = redditor.new(limit=commentParentMax)
                case 'controversial':
                    api_return = redditor.controversial(time_filter=timeframe, limit=commentParentMax)
                case _:
                    print('Invalid stream type')
                    return None

        elif phraseSearch == 'y':
            api_return = redditor.comments(phrase, phrasetime_filter=timeframe, sort=hottopnewcont)
        else:
            print('Invalid input')
            return None

    submissions = list(api_return)
    for submission in submissions:
        if len(data) > count:
            break
        comments = submission.comments
        for comment in comments:
            if len(data) > count:
                break
            try:
                createdAt = datetime.datetime.fromtimestamp(comment.created_utc)
                data.append([comment.body, createdAt, None, comment.ups, None])
            except:
                continue
    return pd.DataFrame(data, columns=columns)



