
def load_sentiment_emotion_data(path):
    """
    It takes a path to a directory of csv files, reads them in, and drops and renames some columns
    
    Args:
      path: the path to the folder containing the csv files for each brand
    
    Returns:
      An appended dataframe with the preprocessed columns
    """

    
def get_hashtags(text):
    """
    Extracts hashtags from text after performing text preprocessing like lowercasing, punctuation removal etc.
    
    Args:
      text: the text you want to extract hashtags from
    
    Returns:
      A list of hashtags
    """


def apply_hashtags(df):
    """
    It takes a dataframe as input, and returns a dataframe with a new column called 'hashtags' that
    contains a list of hashtags for each tweet
    
    Args:
      df: the dataframe
    
    Returns:
      A dataframe with a new column called 'hashtags'
    """


def text_clean_yake(text):
    """
    Cleans the text by removing mentions, hashtags, weblinks, websites, punctuation, non alpha numeric, and additional
    whitespaces
    
    Args:
      text: The text to be processed.
    
    Returns:
      A cleaned text
    """


def get_ht_dict(df):
    """
    For each brand, get the hashtags from the twitter and reddit posts
    
    Args:
      df: the dataframe containing the data
    
    Returns:
      A dictionary of hashtags for each brand and stream
    """



def get_kw_yake_dict(df):
    """
    For each brand, get the top 20 keywords from the YAKE keyword extractor for both Twitter and Reddit
    
    Args:
      df: dataframe with the text data
    
    Returns:
      A dictionary of YAKE keywords for each brand and stream
    """



def spacy_noun_chunks(doc_text, model = 'en_core_web_lg'):
    """
    Extracts SpaCy noun chunks from a given preprocessed document text.
    
    Let's try it out on the first document in the corpus
    
    Args:
      doc_text: the text you want to extract noun chunks from
      model: the model to use for the NLP. Defaults to en_core_web_lg
    
    Returns:
      A list of noun chunks
    """

def get_noun_chunks_dict(df):
    """
    Extract the SpaCy noun chunks from the cleaned text posts
    
    Args:
      df: the dataframe containing the cleaned text data
    
    Returns:
      A dictionary of SpaCy noun chunks for each brand and stream
    """


def get_kw_kbnc_dict(df, kw_model, noun_chunks):
    """
    For each brand, extract the top 20 KeyBERT keywords from the text of each stream (Twitter and Reddit) using
    the noun chunks as candidates
    
    Args:
      df: the dataframe with the text data
      kw_model: the keyword extractor model
      noun_chunks: a dictionary of noun chunks for each brand and stream
    
    Returns:
      A dictionary of keyBERT keywords for each brand and stream
    """
