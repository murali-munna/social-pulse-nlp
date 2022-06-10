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



