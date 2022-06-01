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
    
    brands = ['Iphone', 'Samsung', 'Robinhood', 'WhatsApp', 'Netflix', 'Disney', 'McDonalds', 'Ferrari']
    streams = ['Reddit','Twitter']
    topic_types = ['Hashtags', 'Broad Topics']
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
        
        df = pd.read_pickle(os.path.join(path,'df.pkl'))
        df['date_time'] = pd.to_datetime(df['time_of_creation'])
        df['date'] = pd.to_datetime(df['time_of_creation']).dt.date
        df['days_diff'] = (pd.Timestamp.now() - df['date_time']).dt.days
        df['time_period'] = np.where(df['days_diff']<=7, 1,
                             np.where(df['days_diff']<=31, 2,
                             np.where(df['days_diff']<=180, 3,
                             np.where(df['days_diff']<=365, 4,
                             np.where(df['days_diff']<=365*5, 5, 6)))))
        df['posts'] = 1
        df['votes'] = df['ups'].fillna(df['favorite_count'])
        df['sentiment'] = df['sentiment_prediction'].replace({1:'Positive', 0:'Neutral', -1:'Negative'})
        df['sentiment'] = pd.Categorical(df['sentiment'], ["Negative", "Neutral", "Positive"])
        df = df.rename({'emotion prediction': 'emotion'}, axis=1)
        df['emotion'] = pd.Categorical(df['emotion'], ['joy', 'sadness', 'surprise', 'anger', 'fear'])
        
        kw_yake = pickle.load(open(os.path.join(path,'kw_yake.pkl'), 'rb'))
        # noun_chunks = pickle.load(open(os.path.join(path,'noun_chunks.pkl'), 'rb'))
        kw_kbnc = pickle.load(open(os.path.join(path,'kw_kbnc.pkl'), 'rb'))

        return df, kw_yake, kw_kbnc
    
    @st.experimental_memo
    def get_filters(df):
        return None

    @st.experimental_memo
    def plot_timeline(df, brand, stream, time_period, weight):
        
        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        # long_df = px.data.medals_long()
        # fig = px.bar(long_df, x="nation", y="count", color="medal", title="Long-Form Input")
        
        data = data.groupby(['date','sentiment'])[agg_col].sum().reset_index().sort_values(['date', 'sentiment'], ascending=[True, True])

        fig = px.bar(data, x="date", y=agg_col, color="sentiment", 
                    color_discrete_map={'Negative': colors[0], 'Neutral': colors[1], 'Positive': colors[2]})
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin_l=10, margin_r=10, margin_t=10, margin_b=10)
        # fig['layout']['xaxis']['dtick']=1
        
        return fig
    
    @st.experimental_memo
    def get_filters(df):
        return None

    @st.experimental_memo
    def plot_sentiment(df, brand, stream, time_period, weight):
        
        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        # fig2 = px.pie(pd.DataFrame({'sentiment':['Positive', 'Neutral', 'Negative'], 'score':[30, 60, 10]}), 
        #               values='score', names='sentiment', hole=0.4) #, title=f'<b>Sentiment Distrbution<b>'
        
        data = data.groupby(['sentiment'])[agg_col].sum().reset_index().sort_values(['sentiment'], ascending=[True])

        fig = px.pie(data, values=agg_col, names='sentiment', hole=0.4,
                     color='sentiment', color_discrete_sequence=colors)
        
        # fig.update_layout(legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="right",
        #     x=1
        # ))
        fig.update_traces(sort=False) 
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=25, margin_r=25, margin_t=25, margin_b=25, 
            # yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    @st.experimental_memo
    def plot_wordcloud(df, brand, stream, time_period, weight, ngram):
        
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
        
        # words = []
        # for word, val in ngram_counts.items():
        #     word_color = colors[1] #neutral words' color
        #     sentiment = 'Neutral'
        #     if word in pos_words:
        #         word_color = colors[2] # positive words' color
        #         sentiment = 'Positive'
        #     elif word in neg_words:
        #         word_color = colors[0] # negative words' color
        #         sentiment = 'Negative'
                
        #     dic = dict(text=word, value=val, color=word_color, sentiment=sentiment)
        #     words.append(dic)
        # print(words)
        
        # obj = wordcloud.visualize(words, width='100%', tooltip_data_fields={
        #     'text':'Word', 'value':'Count', 'sentiment':'Sentiment'}, per_word_coloring=True, max_words = 50 if ngram==2 else 100)
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
        
        agg_col = 'posts' if weight=='# Posts' else 'votes'
        data = df[(df['brand']==brand) & (df['stream']==stream) & (df['time_period']<=times_dict[time_period])]
        
        # fig2 = px.pie(pd.DataFrame({'sentiment':['Positive', 'Neutral', 'Negative'], 'score':[30, 60, 10]}), 
        #               values='score', names='sentiment', hole=0.4) #, title=f'<b>Sentiment Distrbution<b>'
        
        data = data.groupby(['emotion'])[agg_col].sum().reset_index().sort_values(['emotion'], ascending=[True])
        
        fig = px.line_polar(data, r=agg_col, theta='emotion', line_close=True, hover_name='emotion')
        fig.update_traces(fill='toself', line_color=colors[3])
        # fig.update_layout(legend=dict(
        #     orientation="h",
        #     yanchor="bottom",
        #     y=1.02,
        #     xanchor="right",
        #     x=1
        # ))
        fig.update_layout(
            # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",x=1),
            showlegend=False,
            margin_l=25, margin_r=25, margin_t=25, margin_b=25, 
            # yaxis_visible=False, yaxis_showticklabels=False
        )
        
        return fig
    
    
    def prepare_topic_data(df, brand, stream, topic_type, weight):
        
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
                x = data['text'].str.lower().str.contains(topic) * data[agg_col].astype(int)
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
        # print(topic_df)
        # print(topic_df['topic'].tolist())
        
        return topic_df
    
    # @st.experimental_memo
    def plot_topic_freq(df, brand, stream, topic_type, weight):
        
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
        
      
    def plot_topic_sentiment(df, brand, stream, topic_type, weight):
        
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
    
    
    def plot_topic_emotion(df, brand, stream, topic_type, weight):
        
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
        
        kw = kw_kbnc[brand][stream]
        
        count_model = CountVectorizer(ngram_range=(1,1)) # default unigram model
        X = count_model.fit_transform(kw)
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
        
        agg_col = 'posts' if weight=='# Posts' else 'votes'
        detect_col = 'sentiment' if detection=='Sentiment' else 'emotion'
        data = df[(df['brand']==brand) & (df['stream']=='Twitter') & (df['time_period']<=times_dict[time_period])]
        
        # fig2 = px.pie(pd.DataFrame({'sentiment':['Positive', 'Neutral', 'Negative'], 'score':[30, 60, 10]}), 
        #               values='score', names='sentiment', hole=0.4) #, title=f'<b>Sentiment Distrbution<b>'
        
        def getLoc(locStr, i):
            if not isinstance(locStr, str):
                return None
            r = locStr.split(',')
            if len(r) < 2:
                return None
            return float(r[i])
        
        data['lat'] = data.apply(lambda row: getLoc(row['location'], 0), axis=1)
        data['lon'] = data.apply(lambda row: getLoc(row['location'], 1), axis=1)
        
        # data['lat_lon'] = data['lat'].astype(str) + '_' + data['lon'].astype(str)
        # data = data.groupby(['lat_lon', detect_col])[agg_col].sum().reset_index()
        # data1 = data.groupby('lat_lon')[agg_col].sum().reset_index()
        # data = pd.merge(data, data1, on='lat_lon')
        # data[agg_col] = data[agg_col+'_x']/data[agg_col+'_y'] * 100
        # data['lat'] = data['lat_lon'].str.split('_', expand=True)[0].astype(float)
        # data['lon'] = data['lat_lon'].str.split('_', expand=True)[1].astype(float)
        # data = data[data[detect_col]==feeling][['lat', 'lon', agg_col]]
        
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
    
    
    df, kw_yake, kw_kbnc = load_data()
    
    # ==================================================================================
    # ========== Streamlit App Code ====================================================
    # ==================================================================================
    
    
    with open('streamlit_app/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

    # padding_top = -20

    # st.markdown(f"""
    #     <style>
    #         .reportview-container .main .block-container{{
    #             padding-top: {padding_top}rem;
    #         }}
    #     </style>""",
    #     unsafe_allow_html=True,
    # )
    # st.markdown(
    #         f'''
    #         <style>
    #             .reportview-container .sidebar-content {{
    #                 padding-top: {1}rem;
    #             }}
    #             .reportview-container .main .block-container {{
    #                 padding-top: {1}rem;
    #             }}
    #         </style>
    #         ''',unsafe_allow_html=True)

    # set_page_container_style(
    #     max_width = 1100, max_width_100_percent = True,
    #     padding_top = 1, padding_right = 1, padding_left = 1, padding_bottom = 1
    # )

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
            # st.write('**Posts Timeline**')
            # st.bar_chart(chart_data, width=200, height=300)
            with st.expander("Posts Timeline", expanded=True):
                st.plotly_chart(plot_timeline(df, brand, stream, time_period, weight), use_container_width=True)
            # source = data.barley()

            # st.altair_chart(
            #     alt.Chart(source).mark_bar().encode(
            #     x='variety',
            #     y='sum(yield)',
            #     color='site'
            #     ),
            #     use_container_width=True
            # )
            
            
        with row2_2:
            # st.write('**Sentiment Distribution**')
            with st.expander("Sentiment Distribution", expanded=True):
                st.plotly_chart(plot_sentiment(df, brand, stream, time_period, weight), use_container_width=True)
            # source = pd.DataFrame({"category": [1, 2, 3, 4, 5, 6], "value": [4, 6, 10, 3, 7, 8]})
            # st.altair_chart(
            #     alt.Chart(source).mark_arc(innerRadius=50).encode(
            #         theta=alt.Theta(field="value", type="quantitative"),
            #         color=alt.Color(field="category", type="nominal"),
            #     ),
            #     use_container_width=True
            # )
        
        row3_1, row3_2 = st.columns([3,2])
        with row3_1:
            with st.expander("N-gram Word Cloud", expanded=True):
                ngram = st.radio(f'N-gram', ['uni-gram', 'bi-gram'], index=1)
                # st.write(plot_wordcloud(df, brand, stream, time_period, weight, ngram), use_container_width=True)
                # 1 if ngram=='uni-gram' else 2
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
                -   **Sentiment Detection**:
                -   **Emotion Detection**:
                -   **Keyword Extraction**:
                """
        )
        
        with st.expander("Documentation", expanded=True):
            st.write(
                """     
                -   **Github**: [Social Pulse](https://github.com/murali-munna/social-pulse-nlp)
                -   **Read the Docs**: 
                """
        )
            
        with st.expander("Team", expanded=True):
            st.write(
                """     
                -   Murali Dandu, ...
                """
        )
    
    

if __name__ == '__main__':
    main()
    


