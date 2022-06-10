

def TwitterRoutine(Redd_Twitt_queries, sname):
    """
    This function takes a list of search queries and a name of a social media platform, and returns a dataframe of tweets
    that match the search queries

    Args:
      Redd_Twitt_queries: a list of lists, where each list contains a list of search terms and a list of search phrases.
      sname: the name of the subreddit

    Returns:
      A dataframe with the columns:
        - query
        - source
        - text
        - date
        - location
        - subreddit
        - likes
    """



def RedditRoutine(Redd_Twitt_queries, sname):
    """
    This function takes in a list of search queries and a search name, and returns a dataframe of Reddit comments

    Args:
      Redd_Twitt_queries: A list of lists of search terms. The first list is for Reddit, the second for Twitter.
      sname: The name of the stock you want to search for.

    Returns:
      A dataframe with the following columns:
        - Topic
        - Source
        - Text
        - CreatedAt
        - RetweetCount
        - FavoriteCount
        - Hashtags
    """

def autoScrape(industryDict):
    """
    This function takes a dictionary of industry names and their associated queries, and then runs the Twitter and Reddit
    routines on each query

    Args:
      industryDict: a dictionary of industry names and their corresponding queries

    Returns:
      A dataframe with the scraped data.
    """
