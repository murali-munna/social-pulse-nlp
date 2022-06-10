import os
import pandas as pd
import numpy as np
import streamlit as st
# import streamlit_wordcloud as wordcloud
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
# from common import set_page_container_style
import matplotlib.pyplot as plt
import plotly.express as px
# import altair as alt
# from vega_datasets import data
import pickle
import re
import nltk
# nltk.download('stopwords')
# nltk.downloader.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from imageio.v2 import imread
from wordcloud import WordCloud
from urllib.error import URLError
import pydeck as pdk
from show_extreme_posts import postExtremeComments
import networkx as nx
from pyvis import network as net


def main():
    
    st.set_page_config(
            page_title="Social Pulse",
            # page_icon="chart_with_upwards_trend",
            layout="wide",
    )
    
    # ==================================================================================
    # ========== Data ==================================================================
    # ==================================================================================
    
    path = 'streamlit_app/data/'
    
    brands = ['Iphone', 'Samsung', 'GooglePixel', 'Dell', 'Microsoft', 'Facebook', 'Robinhood', 'Tesla', 'Ferrari', 'WhatsApp', 'Netflix', 'Disney', 'McDonalds']
    streams = ['Reddit','Twitter']
    topic_types = ['Broad Topics', 'Hashtags']
    times = ['Last 1w', 'Last 1m', 'Last 6m', 'Last 1yr', 'Last 5yr']
    times_dict = {'Last 1w':1, 'Last 1m':2, 'Last 6m':3, 'Last 1yr':4, 'Last 5yr':5}
    weightages = ['# Posts', '# Votes']
    sent_feelings = ['Negative', 'Neutral', 'Positive']
    emo_feelings = ['joy', 'sadness', 'surprise', 'anger', 'fear']
    colors = ['#e86252', '#ddb967', '#43aa8b', '#086788']
    posColor = 'rgb(128,177,211)'
    negColor = 'rgb(128,177,211)'
    quantity = 10
    keywordMarker = 'KW'
    probThresh = 0.99
    
    # load data once
    @st.experimental_memo
    def load_data():
        """
        It loads the dataframe, the hashtags, the keywords from the Yake algorithm, and the keywords from the noun chunks.

        Returns:
          df, kw_ht, kw_yake, kw_kbnc
        """

        df = pd.read_pickle(os.path.join(path,'df.pkl'))
        
        kw_ht = pickle.load(open(os.path.join(path,'kw_ht.pkl'), 'rb'))
        kw_yake = pickle.load(open(os.path.join(path,'kw_yake.pkl'), 'rb'))
        # noun_chunks = pickle.load(open(os.path.join(path,'noun_chunks.pkl'), 'rb'))
        kw_kbnc = pickle.load(open(os.path.join(path,'kw_kbnc.pkl'), 'rb'))

        return df, kw_ht, kw_yake, kw_kbnc



    @st.experimental_memo
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

        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        data = data.groupby(['date','sentiment'])[agg_col].sum().reset_index().sort_values(['date', 'sentiment'], ascending=[True, True])

        fig = px.bar(data, x="date", y=agg_col, color="sentiment", 
                    color_discrete_map={'Negative': colors[0], 'Neutral': colors[1], 'Positive': colors[2]})
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin_l=10, margin_r=10, margin_t=10, margin_b=10)
        
        return fig
    

    @st.experimental_memo
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

        
        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        data = data.groupby(['sentiment'])[agg_col].sum().reset_index().sort_values(['sentiment'], ascending=[True])

        fig = px.pie(data, values=agg_col, names='sentiment', hole=0.4,
                     color='sentiment', color_discrete_sequence=colors)
        
        fig.update_traces(sort=False) 
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=55, margin_r=55, margin_t=55, margin_b=55, 
            # yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    @st.experimental_memo
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

        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        STOP_WORDS = stopwords.words('english')
        
        def clean_text(text):
            brand_lower = brand.lower()
            temp = text.lower()
            if brand.lower() == 'iphone':
                temp = re.sub(brand_lower + ' [0-9]+', '', temp) # for iphone model
                temp = re.sub(brand_lower + '[0-9]+', '', temp) # for iphone model
                temp = re.sub('apple', '', temp)
            temp = re.sub(brand_lower, ' ', temp)
            temp = re.sub("'", "", temp) # to avoid removing contractions in english
            temp = re.sub("@[A-Za-z0-9_]+","", temp) # mentions
            temp = re.sub("#[A-Za-z0-9_]+","", temp) # hashtags
            temp = re.sub(r"http\S+", "", temp) # weblinks
            temp = re.sub(r"www.\S+", "", temp) # websites
            temp = re.sub('[()!?]', ' ', temp) # punctuation
            temp = re.sub('\[.*?\]',' ', temp)
            temp = re.sub('\.\Z', '', temp) # remove dot but not remove decimal point
            temp = re.sub('\.\s+', ' ', temp) # remove dot but not remove decimal point
            reg_float='^\d+\.\d+$'
            temp = re.sub("[^a-z0-9.]"," ", temp) # non alpha numeric dot
            temp = temp.split()
            temp = [w for w in temp if not w in STOP_WORDS]
            temp = " ".join(word for word in temp)
            return temp
        
        texts = data['text'].apply(clean_text).tolist()
        # texts = ' '.join([x.strip()+'.' for x in texts])
        count_vect = CountVectorizer(ngram_range=(ngram,ngram))
        count_vect.fit(texts)
        ngram_counts = count_vect.vocabulary_
        # print(ngram_counts)
        
        ngram_sia = {}
        sia = SentimentIntensityAnalyzer()
        for r in ngram_counts.keys():
            if not r.lower() in STOP_WORDS:
                neg_polarity = sia.polarity_scores(r)['neg']
                pos_polarity = sia.polarity_scores(r)['pos']
                cpd_polarity = sia.polarity_scores(r)['compound']
                sentiment = (neg_polarity+pos_polarity)*cpd_polarity
                if(sentiment != 0):
                    ngram_sia[r] = sentiment
        
        pos_words = []
        neg_words = []
        for key, value in ngram_sia.items():
            if(value > 0):
                pos_words.append(key)
            else:
                neg_words.append(key)
        
        
        class AssignColour(object):
            def __init__(self, color_to_words, default_color):
                self.word_to_color = {word: color
                                    for (color, words) in color_to_words.items()
                                    for word in words}

                self.default_color = default_color

            def __call__(self, word, **kwargs):
                return self.word_to_color.get(word, self.default_color)
        
        colour_words_dict = {
            colors[2]: pos_words,
            colors[0]: neg_words
        }
        
        logo = 'twitter_logo.png' if stream=='Twitter' else 'reddit_logo.png'
        twitter_logo = imread(os.path.join('streamlit_app/data/logo',logo))

        grouped_colour_func = AssignColour(colour_words_dict, colors[1])

        twitter_wc = WordCloud(mask = twitter_logo, width=200, height=100, 
                            max_words = 1000, collocations = False, background_color = 'white', 
                            color_func=grouped_colour_func).generate_from_frequencies(ngram_counts) 
        
        fd = {
            'fontsize': '32',
            'fontweight' : 'normal',
            'verticalalignment': 'baseline',
            'horizontalalignment': 'center',
        }
        plt.figure()
        fig, ax = plt.subplots(figsize=(3,2))
        ax.imshow(twitter_wc, interpolation='bilinear')
        ax.axis('off')
        # ax.set_title(f'{ngram_name} Words of twitter', pad=24, fontdict=fd)
        
        return fig
    
    @st.experimental_memo
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

        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        
        data = data.groupby(['emotion'])[agg_col].sum().reset_index().sort_values(['emotion'], ascending=[True])
        
        fig = px.line_polar(data, r=agg_col, theta='emotion', line_close=True, hover_name='emotion')
        fig.update_traces(fill='toself', line_color=colors[3])
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=45, margin_r=45, margin_t=45, margin_b=45, 
            # yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    
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
        

        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream)]
        
        if topic_type=='Hashtags':
            topics = list(set([ht for ht_set in data['hashtags'].tolist() for ht in ht_set if ht_set]))
        elif topic_type=='Broad Topics':
            topics = kw_yake[brand][stream]
        else:
            topics = kw_kbnc[brand][stream]
        
        topic_count = []
        for topic in topics:
            if topic_type=='Hashtags':
                # x = data['text'].str.lower().str.contains(topic) * data[agg_col].astype(int)
                x = data['text'].apply(lambda li: topic in li) * data[agg_col].astype(int)
            else:
                x = data['text_clean_yake'].str.contains(topic) * data[agg_col].astype(int)
            topic_count.append([
                topic,
                ((x) & (data['sentiment']=='Negative')).sum(),
                ((x) & (data['sentiment']=='Neutral')).sum(),
                ((x) & (data['sentiment']=='Positive')).sum(),
                ((x) & (data['emotion']=='joy')).sum(),
                ((x) & (data['emotion']=='sadness')).sum(),
                ((x) & (data['emotion']=='surprise')).sum(),
                ((x) & (data['emotion']=='anger')).sum(),
                ((x) & (data['emotion']=='fear')).sum(),
            ])
        
        topic_df = pd.DataFrame(topic_count, columns=['topic', 'Negative', 'Neutral', 'Positive',
                                   'joy', 'sadness', 'surprise', 'anger', 'fear'])
        topic_df['total'] = topic_df[['Negative', 'Neutral', 'Positive']].sum(axis=1)
        topic_df = topic_df.sort_values('total', ascending=False).iloc[:15,:].iloc[::-1]
        
        return topic_df
    
    @st.experimental_memo
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

        topic_df = prepare_topic_data(df, brand, stream, topic_type, weight)  
        
        fig = px.bar(topic_df, y='topic', x='total', orientation='h')
        fig.update_traces(marker_color='#086788')
        
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=10, margin_r=10, margin_t=10, margin_b=10, 
            # yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
        
    @st.experimental_memo
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

        topic_df = prepare_topic_data(df, brand, stream, topic_type, weight)
        
        topic_df_sent = topic_df[['topic','Negative', 'Neutral', 'Positive']].set_index('topic')
        topic_df_sent = topic_df_sent.div(topic_df_sent.sum(axis=1), axis=0).reset_index()
        topic_df_sent = pd.melt(topic_df_sent, id_vars='topic', 
                                value_vars=['Negative', 'Neutral', 'Positive'])
        topic_df_sent.columns = ['topic', 'sentiment', 'value']
        fig = px.bar(topic_df_sent, x="value", y="topic", color="sentiment", orientation='h',
                    color_discrete_map={'Negative': colors[0], 'Neutral': colors[1], 'Positive': colors[2]},
                    hover_data={'value':':.1%', # remove species from hover data
                                    })
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=10, margin_r=10, margin_t=10, margin_b=10, 
            yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    @st.experimental_memo
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

        topic_df = prepare_topic_data(df, brand, stream, topic_type, weight)
        
        topic_df_sent = topic_df[['topic']+emo_feelings].set_index('topic')
        topic_df_sent = topic_df_sent.div(topic_df_sent.sum(axis=1), axis=0).reset_index()
        topic_df_sent = pd.melt(topic_df_sent, id_vars='topic', 
                                value_vars=emo_feelings)
        topic_df_sent.columns = ['topic', 'emotion', 'value']
        fig = px.bar(topic_df_sent, x="value", y="topic", color="emotion", orientation='h',
                    # color_discrete_map={'joy': 'rgb(102, ', 'sadness': colors[1], 'surprise': colors[2], 'anger':, 'fear':},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    hover_data={'value':':.1%', # remove species from hover data
                                    })
        fig.update_layout(
            # legend=dict(orientation="h"),
            showlegend=False,
            margin_l=10, margin_r=10, margin_t=10, margin_b=10, 
            yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    def show_network(brand, stream, kw_kbnc):
        """
        It takes in a brand, stream, and a dictionary of keywords, and outputs a network graph of the keywords.

        Args:
          brand: the brand you want to look at
          stream: 'all', 'news', 'twitter', 'reddit', 'blogs', 'forums', 'reviews', 'image', 'video'
          kw_kbnc: a dictionary of dictionaries, where the first key is the brand, the second key is the stream, and the
        value is a list of keywords.
        """

        kw = kw_kbnc[brand][stream]
        
        STOP_WORDS = stopwords.words('english')
        kw_clean = []
        for k in kw:
            s = [w for w in k.split() if w not in STOP_WORDS]
            kw_clean.append(' '.join(s))
            
            
        count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
        X = count_model.fit_transform(kw_clean)
        # X[X > 0] = 1 # run this line if you don't want extra within-text cooccurence (see below)
        Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
        Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
        
        vocab_map = {v:k for k,v in count_model.vocabulary_.items()}
        
        G = nx.from_numpy_matrix(Xc.todense())
        G = nx.relabel_nodes(G, vocab_map)
        scale=5 # Scaling the size of the nodes by 10*degree
        d = dict(G.degree)
        #Updating dict
        d.update((x, scale*y) for x, y in d.items())
        #Setting up size attribute
        nx.set_node_attributes(G,d,'size')
        
        # g = net.Network()
        g = net.Network(height='500px', width='90%')
        g.from_nx(G)
        # g.barnes_hut(central_gravity=0.8)
        # g.show_buttons(filter_=['physics'])
        g.show(os.path.join(path, 'kw_graph.html'))
        
        HtmlFile = open(os.path.join(path, 'kw_graph.html'), 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height = 1200, width=1200)
    
    
    # @st.experimental_memo
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

        agg_col = 'posts' if weight=='# Posts' else 'votes'
        detect_col = 'sentiment' if detection=='Sentiment' else 'emotion'
        data = df[(df['brand']==brand) & (df['stream']=='Twitter') & (df['time_period']<=times_dict[time_period])]
        
        
        def getLoc(locStr, i):
            if not isinstance(locStr, str):
                return None
            r = locStr.split(',')
            if len(r) < 2:
                return None
            return float(r[i])
        
        data['lat'] = data.apply(lambda row: getLoc(row['location'], 0), axis=1)
        data['lon'] = data.apply(lambda row: getLoc(row['location'], 1), axis=1)

        
        data = data[data[detect_col]==feeling][['lat', 'lon', detect_col, agg_col]]
        
        if agg_col=='votes':
            # print(data)
            new_data=data.reindex(data.index.repeat(data[agg_col]))
            new_data['position']=new_data.groupby(level=0).cumcount()+1
            # print(new_data)
        else:
            # print(data)
            new_data = data
        
        return new_data
    
    # @st.experimental_memo(suppress_st_warning=True)
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

        new_data = prepare_geo_data(df, brand, time_period, weight, detection, feeling)
        
        try:
            ALL_LAYERS = {
                'Feeling': pdk.Layer(
                    "HexagonLayer",
                    data=new_data,
                    get_position=["lon", "lat"],
                    # get_elevation=agg_col,
                    radius=80000,
                    elevation_scale=4,
                    elevation_range=[300000, 600000],
                    extruded=True,
                ),

            }
            # st.sidebar.markdown('### Map Layers')
            tooltip = {
                "html": "<b>{lat},{lon}</b>",
                "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
            }
            selected_layers = [
                layer for layer_name, layer in ALL_LAYERS.items()
                # if st.sidebar.checkbox(layer_name, True)
                ]  # all selected
            if selected_layers:
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/light-v9",
                    initial_view_state={"latitude": 37.4214,
                                        "longitude": -100, "zoom": 2.2, "pitch": 50},
                    tooltip=tooltip,
                    layers=selected_layers,
                ))
            else:
                st.error("Please choose at least one layer above.")
        except URLError as e:
            st.error("""**This demo requires internet access.**Connection error: %s""" % e.reason)
        
        # view = pdk.data_utils.compute_view(data[["lon", "lat"]])
        # view.pitch = 75
        # view.bearing = 60
        
        # column_layer = pdk.Layer(
        #     "ColumnLayer",
        #     data=data,
        #     get_position=["lng", "lat"],
        #     get_elevation=agg_col,
        #     elevation_scale=100,
        #     radius=50,
        #     # get_fill_color=["mrt_distance * 10", "mrt_distance", "mrt_distance * 10", 140],
        #     pickable=True,
        #     auto_highlight=True,
        # )

        # tooltip = {
        #     "html": "<b>{mrt_distance}</b> meters away from an MRT station, costs <b>{price_per_unit_area}</b> NTD/sqm",
        #     "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
        # }

        # r = pdk.Deck(
        #     column_layer,
        #     initial_view_state={"latitude": 37.4214, "longitude": -100, "zoom": 2, "pitch": 50},
        #     # tooltip=tooltip,
        #     map_provider="mapbox",
        #     map_style="mapbox://styles/mapbox/light-v9",
        # )
        
        # st.pydeck_chart(r)
    
    
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

        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]

        keywords = kw[brand][stream]

        positiveData = data[(data["sentiment_prediction"] == 1) & (data['sentiment_probability'] > probThresh)]\
            .sort_values(by=["votes","sentiment_probability"], ascending=[False,False])
        negativeData = data[(data["sentiment_prediction"] == -1) & (data['sentiment_probability'] > probThresh)]\
            .sort_values(by=["votes","sentiment_probability"], ascending=[False,False])
        # negativeData = negativeData_raw[negativeData_raw['sentiment_probability'] < negProbThresh]

        topPos = positiveData.iloc[0:quantity]
        topNeg = negativeData.iloc[0:quantity]

        # """
        # # Most positive comments:
        # """
        st.header('Extreme Comments')
        postExtremeComments(topPos, keywords, posColor, extreme='Positive')
        # """
        
        # # Most negative comments:
        # """
        postExtremeComments(topNeg, keywords, negColor, extreme='Negative')
    
    
    df, kw_ht, kw_yake, kw_kbnc = load_data()
    
    # ==================================================================================
    # ========== Streamlit App Code ====================================================
    # ==================================================================================
    
    
    with open('streamlit_app/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

    st.title("💬 Social Pulse")


    with st.sidebar:
        choose_menu = option_menu("App Gallery", ["Overview", "Keyword Analytics", "Geo Analytics", "Data Deepdive", "About"],
                            icons=['twitter', 'chat-left-text-fill', 'geo-alt-fill', 'table', 'info-circle-fill'],
                            menu_icon="app-indicator", default_index=0,
        #                      styles={
        #     "container": {"padding": "5!important", "background-color": "#fafafa"},
        #     "icon": {"color": "orange", "font-size": "25px"}, 
        #     "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        #     "nav-link-selected": {"background-color": "#02ab21"},
        # }
        orientation='horizantal'
        )
    
    
    if choose_menu=='Overview':
        
        row1_1, _, row1_2, row1_3, _, row1_4 = st.columns([1, 0.2, 1, 1, 0.2, 1])
        with row1_1:
            brand = st.selectbox(f'Select Brand', brands, help='Choose the brand that you want to analyze. The current prototype contains 10+ brands')
        with row1_2:
            stream = st.radio(f'Select Social Stream', streams, help='Choose the social media')
        with row1_3:
            time_period = st.select_slider(f'Select Time Period', times, value='Last 1m', help='Choose the time period')
        with row1_4:
            weight = st.radio(f'Weightage Scheme', weightages, help='Choose if you want to weight the values by upvotes')

        
        row2_1, row2_2 = st.columns([3,2])
        with row2_1:
            with st.expander("Posts Timeline", expanded=True):
                st.plotly_chart(plot_timeline(df, brand, stream, time_period, weight), use_container_width=True)
            
            
        with row2_2:
            with st.expander("Sentiment Distribution", expanded=True):
                st.plotly_chart(plot_sentiment(df, brand, stream, time_period, weight), use_container_width=True)
                
        
        row3_1, row3_2 = st.columns([3,2])
        with row3_1:
            with st.expander("N-gram Word Cloud", expanded=True):
                ngram = st.radio(f'N-gram', ['uni-gram', 'bi-gram'], index=1)
                st.pyplot(plot_wordcloud(df, brand, stream, time_period, weight, 1 if ngram=='uni-gram' else 2))
            
        with row3_2:
            with st.expander("Emotion Distribution", expanded=True):
                st.plotly_chart(plot_emotion(df, brand, stream, time_period, weight), use_container_width=True)
    
    if choose_menu=='Keyword Analytics':

        row1_1, _, row1_2, row1_3, _, row1_4, _ = st.columns([1, 0.1, 1, 1, 0.1, 1, 1])
        with row1_1:
            brand = st.selectbox(f'Select Brand', brands, help='Choose the brand that you want to analyze. The current prototype contains 10+ brands')
        with row1_2:
            stream = st.radio(f'Select Social Stream', streams, help='Choose the social media')
        with row1_3:
            topic_type = st.selectbox(f'Select Topic Type', topic_types if stream=='Twitter' else ['Broad Topics'], help='Choose the the type/level of topic to analyze')
        with row1_4:
            weight = st.radio(f'Weightage Scheme', weightages, help='Choose if you want to weight the values by upvotes')

        row2_1, row2_2, row2_3 = st.columns([4,3,3])
        with row2_1:    
            with st.expander("Topic Frequency", expanded=True):
                # st.plotly_chart(plot_timeline(df, brand, stream, time_period, weight), use_container_width=True)
                st.plotly_chart(plot_topic_freq(df, brand, stream, topic_type, weight), use_container_width=True)
            
        with row2_2:
            with st.expander("Topic Sentiment", expanded=True):
                # st.plotly_chart(plot_sentiment(df, brand, stream, time_period, weight), use_container_width=True)
                st.plotly_chart(plot_topic_sentiment(df, brand, stream, topic_type, weight), use_container_width=True)
        
        with row2_3:
            with st.expander("Topic Emotion", expanded=True):
                # st.plotly_chart(plot_sentiment(df, brand, stream, time_period, weight), use_container_width=True)
                st.plotly_chart(plot_topic_emotion(df, brand, stream, topic_type, weight), use_container_width=True)
        
        st.subheader("Keyphrase Network")
        show_network(brand, stream, kw_kbnc)
    
    if choose_menu=='Geo Analytics':
        
        row1_1, _, row1_3, row1_4, row1_5, row1_6 = st.columns([1, 0.1, 1, 1, 1, 1])
        with row1_1:
            brand = st.selectbox(f'Select Brand', brands, help='Choose the brand that you want to analyze. The current prototype contains 10+ brands')
        # with row1_2:
        #     stream = st.radio(f'Select Social Stream', streams, help='Choose the social media')
        with row1_3:
            time_period = st.select_slider(f'Select Time Period', times, value='Last 1m', help='Choose the time period')
        with row1_4:
            weight = st.radio(f'Weightage Scheme', weightages, help='Choose if you want to weight the values by upvotes')
        with row1_5:
            detection = st.radio(f'Select Detection', ['Sentiment', 'Emotion'], help='Choose the type of detection')
        with row1_6:
            feeling = st.selectbox(f'Select Brand', sent_feelings if detection=='Sentiment' else emo_feelings, help='Choose the feeling')

        display_geo_analysis(df, brand, time_period, weight, detection, feeling)
     
     
    if choose_menu=='Data Deepdive':
        
        row1_1, _, row1_2, row1_3, _, _ = st.columns([1, 0.2, 1, 1, 0.2, 1])
        with row1_1:
            brand = st.selectbox(f'Select Brand', brands, help='Choose the brand that you want to analyze. The current prototype contains 10+ brands')
        with row1_2:
            stream = st.radio(f'Select Social Stream', streams, help='Choose the social media')
        with row1_3:
            time_period = st.select_slider(f'Select Time Period', times, value='Last 1m', help='Choose the time period')
        
        show_posts(df, brand, stream, time_period, kw_yake)


     
    if choose_menu=='About':
        # st.header('Description')
        with st.expander("Description", expanded=True):
            st.write(
                """     
                -   This app analyzes the brand social media data in terms of sentiment, emotion, keywords and more!
                -   A **Brand Manager** can ask and answer several questions like:
                    - How is my brand's sentiment over a specific time period?
                    - How is the user perception of the new features launched?
                    - What are the key-topics that users are talking about and how's their sentiment?
                    - How is the sentiment/perception varying across geographies and how can I allocate my budget and personnel accordingly?
                    - How are Twitter users different to Reddit and how can I perform targeting marketing based on that?
                    - How is my competitor brand performing?
                """
        )
            
        with st.expander("Data and Models", expanded=True):
            st.write(
                """     
                -   **Twitter Data**: Scraped relevant hashtags using [Tweepy](https://www.tweepy.org/)
                -   **Reddit Data**: Scraped relevant sub-reddits and search terms using [PRAW](https://praw.readthedocs.io/en/stable/)
                
                
                -   **Sentiment Detection**: 'Twitter and Reddit Sentimental analysis Dataset' modelled to detect Positive, Neutral and Negative classes. We used LinearSVC model with TF-IDF representations.
                -   **Emotion Detection**: 'Emotions Dataset' modelled to detect five emotions: joy, sadness, surprise, anger, fear. We used LinearSVC model with TF-IDF representations.
                -   **Keyword Extraction**: 
                    - YAKE: Unsupervised keyword extraction method based on text statistical features
                    - KeyBERT: Keyword extraction technique based on similarity between phrases (noun chunks) and document embeddings
                """
        )
        
        with st.expander("Documentation", expanded=True):
            st.write(
                """     
                -   **Github**: [Social Pulse](https://github.com/murali-munna/social-pulse-nlp)
                -   **Documentation**: 
                """
        )
            
        with st.expander("Team", expanded=True):
            st.write(
                """     
                - Murali Dandu
                - Christopher Light
                - Hao Qu
                - Liyang Ru
                - Pai Tong
                - Yen-Ju Tseng
                """
        )
    
    

if __name__ == '__main__':
    main()
    


