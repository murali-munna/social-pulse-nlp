
def load_data():
    """
    It loads the dataframe, the hashtags, the keywords from the Yake algorithm, and the keywords from the noun chunks.

    Returns:
      df, kw_ht, kw_yake, kw_kbnc
    """


def plot_timeline(df, brand, stream, time_period, weight):
    """
    It takes in a dataframe, brand, stream, time period, and weight, and returns a plotly figure of the timeline of the
    brand's sentiment

    Args:
      df: the dataframe
      brand: the brand you want to look at
      stream: the stream of data you want to look at (e.g. Twitter, Reddit, etc.)
      time_period: The time period to look at.
      weight: # Posts or # Votes

    Returns:
      A figure object
    """


def plot_sentiment(df, brand, stream, time_period, weight):
    """
    It takes in a dataframe, brand, stream, time_period, and weight, and returns a pie chart of the sentiment
    distribution for the given brand, stream, and time_period

    Args:
      df: the dataframe
      brand: Brand name
      stream: 'All', 'Reddit', 'Twitter', 'Facebook', 'Instagram', 'Youtube', 'Forums'
      time_period: The time period for which you want to see the data.
      weight: # Posts or # Votes

    Returns:
      A figure object
    """


def plot_wordcloud(df, brand, stream, time_period, weight, ngram):
    """
    It takes in the dataframe, brand, stream, time period, weight and ngram and returns a wordcloud plot

    Args:
      df: The dataframe that contains the data
      brand: The brand you want to analyze.
      stream: Twitter or Reddit
      time_period: The time period for which you want to see the word cloud.
      weight: The weighting of the word cloud. You can choose between the number of posts or the number of votes.
      ngram: The number of words to consider in a phrase. For example, if ngram is 2, then the word cloud will consider
    2-word phrases.

    Returns:
      A figure object
    """



def plot_emotion(df, brand, stream, time_period, weight):
    """
    It takes in a dataframe, brand, stream, time_period, and weight, and returns a plotly figure of the emotion
    distribution for the given parameters

    Args:
      df: the dataframe
      brand: The brand you want to analyze.
      stream: 'All', 'Twitter', 'Reddit', 'Facebook', 'Instagram'
      time_period: 'All Time', 'Last Year', 'Last Month', 'Last Week'
      weight: '# Posts' or '# Votes'

    Returns:
      A figure object
    """



def prepare_topic_data(df, brand, stream, topic_type, weight):
    """
    It takes a dataframe, a brand, a stream, a topic type, and a weight, and returns a dataframe
    with the top 15 topics for that brand, stream, topic type, and weight

    Args:
      df: the dataframe containing the data
      brand: The brand you want to analyze.
      stream: The stream you want to analyze.
      topic_type: Hashtags, Broad Topics, or Keywords
      weight: '# Posts' or '# Votes'

    Returns:
      A dataframe with the top 15 topics for the given brand, stream, topic type, and weight.
    """



def plot_topic_freq(df, brand, stream, topic_type, weight):
    """
    It takes a dataframe, a brand, a stream, a topic type, and a weight, and returns a plotly figure of the topic
    frequencies for that brand, stream, topic type, and weight

    Args:
      df: the dataframe
      brand: the brand you want to look at
      stream: 'all', 'twitter', 'facebook', 'instagram'
      topic_type: 'topic' or 'subtopic'
      weight: 'weight' or 'count'

    Returns:
      A bar chart of the topic frequencies for the specified brand, stream, topic type, and weight.
    """



def plot_topic_sentiment(df, brand, stream, topic_type, weight):
    """
    It takes a dataframe, a brand, a stream, a topic type, and a weight, and returns a plotly figure of the topic
    sentiment distribution

    Args:
      df: the dataframe containing the data
      brand: the brand you want to analyze
      stream: 'twitter' or 'instagram'
      topic_type: 'lda' or 'nmf'
      weight: 'tfidf' or 'count'

    Returns:
      A plotly figure object
    """



def plot_topic_emotion(df, brand, stream, topic_type, weight):
    """
    It takes in a dataframe, a brand, a stream, a topic type, and a weight, and returns a plotly figure of the topic's
    emotion distribution

    Args:
      df: the dataframe containing the data
      brand: the brand you want to analyze
      stream: 'twitter' or 'instagram'
      topic_type: 'lda' or 'nmf'
      weight: 'tfidf' or 'count'
    """



def show_network(brand, stream, kw_kbnc):
    """
    It takes in a brand, stream, and a dictionary of keywords, and outputs a network graph of the keywords.

    Args:
      brand: the brand you want to look at
      stream: 'all', 'news', 'twitter', 'reddit', 'blogs', 'forums', 'reviews', 'image', 'video'
      kw_kbnc: a dictionary of dictionaries, where the first key is the brand, the second key is the stream, and the
    value is a list of keywords.
    """



def prepare_geo_data(df, brand, time_period, weight, detection, feeling):
    """
    It takes in the dataframe, brand, time period, weight, detection, and feeling, and returns a dataframe with the
    latitude, longitude, and the frequency of the feeling

    Args:
      df: the dataframe
      brand: The brand you want to analyze.
      time_period: The time period for which you want to see the data.
      weight: # Posts or # Votes
      detection: Sentiment or Emotion
      feeling: 'Positive', 'Negative', 'Neutral', 'Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness',
    'Surprise', 'Trust'

    Returns:
      A dataframe with the following columns:
        - lat: latitude of the location
        - lon: longitude of the location
        - sentiment: sentiment of the tweet
        - posts: number of posts
        - position: position of the tweet in the dataframe
    """


def display_geo_analysis(df, brand, time_period, weight, detection, feeling):
    """
    > This function takes in a dataframe, brand, time period, weight, detection, and feeling, and returns a map of the
    world with hexagons representing the number of tweets in that area

    Args:
      df: the dataframe
      brand: the brand you want to analyze
      time_period: the time period you want to look at
      weight: the weight of the sentiment score
      detection: the type of detection you want to use.
      feeling: the feeling you want to analyze
    """



def show_posts(df, brand, stream, time_period, kw):
    """
    It takes in a dataframe, a brand, a stream, a time period, and a dictionary of keywords, and returns a table of the
    most positive and negative comments for that brand, stream, and time period

    Args:
      df: the dataframe containing the data
      brand: the brand you want to look at
      stream: the stream of data you want to look at (e.g. 'twitter', 'reddit', 'news')
      time_period: the time period you want to look at.
      kw: a dictionary of keywords for each brand and stream
    """
