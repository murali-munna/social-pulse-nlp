import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pydeck as pdk
import re
import numpy as np

from urllib.error import URLError


def display_template_geodata():
    try:
        ALL_LAYERS = {
            "Bike Rentals": pdk.Layer(
                "HexagonLayer",
                data=from_data_file("bike_rental_stats.json"),
                get_position=["lon", "lat"],
                radius=200,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
            "Bart Stop Exits": pdk.Layer(
                "ScatterplotLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_color=[200, 30, 0, 160],
                get_radius="[exits]",
                radius_scale=0.05,
            ),
            "Bart Stop Names": pdk.Layer(
                "TextLayer",
                data=from_data_file("bart_stop_stats.json"),
                get_position=["lon", "lat"],
                get_text="name",
                get_color=[0, 0, 0, 200],
                get_size=15,
                get_alignment_baseline="'bottom'",
            ),
            "Outbound Flow": pdk.Layer(
                "ArcLayer",
                data=from_data_file("bart_path_stats.json"),
                get_source_position=["lon", "lat"],
                get_target_position=["lon2", "lat2"],
                get_source_color=[200, 30, 0, 160],
                get_target_color=[200, 30, 0, 160],
                auto_highlight=True,
                width_scale=0.0001,
                get_width="outbound",
                width_min_pixels=3,
                width_max_pixels=30,
            ),
        }
        st.sidebar.markdown('### Map Layers')
        selected_layers = [
            layer for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)]
        if selected_layers:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": 37.76,
                                    "longitude": -122.4, "zoom": 11, "pitch": 50},
                layers=selected_layers,
            ))
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error("""
            **This demo requires internet access.**

            Connection error: %s
        """ % e.reason)

def assignRandomEmotion(df):
    """
    for test only
    """
    emotions = ['happy', 'sad', 'angry']
    df['emotion'] = df.apply(lambda row: np.random.choice(emotions), axis=1)
    return df

def getLoc(locStr, i):
    if not isinstance(locStr, str):
        return None
    r = locStr.split(',')
    if len(r) < 2:
        return None
    return float(r[i])

def preprocess(df):
    emotions = set(df['emotion'])
    for emo in emotions:
        df[emo] = df['emotion'] == emo
        df[emo] = df[emo].astype(int)
    # df = df.groupby(['location'], as_index=False).sum()
    df['lat'] = df.apply(lambda row: getLoc(row['location'], 0), axis=1)
    df['lon'] = df.apply(lambda row: getLoc(row['location'], 1), axis=1)
    df = df[df['lat'].notnull()]
    return df, emotions

def display_geo_analysis():
    df = pd.read_csv('scrapeDF_273_Iphone.csv')
    df = assignRandomEmotion(df)  # ! delete after using real data
    df, emotions = preprocess(df)
    df = df[['lat', 'lon', 'emotion']]
    emotion = st.selectbox(
        "select emotion:", emotions)
    df = df[df['emotion'] == emotion]
    try:
        ALL_LAYERS = {
            "emotions": pdk.Layer(
                "HexagonLayer",
                data=df,
                get_position=["lon", "lat"],
                radius=80000,
                elevation_scale=4,
                elevation_range=[300000, 600000],
                extruded=True,
            ),

        }
        # st.sidebar.markdown('### Map Layers')
        selected_layers = [
            layer for layer_name, layer in ALL_LAYERS.items()
            # if st.sidebar.checkbox(layer_name, True)
            ]  # all selected
        if selected_layers:
            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state={"latitude": 37.4214,
                                    "longitude": -100, "zoom": 2, "pitch": 50},
                layers=selected_layers,
            ))
        else:
            st.error("Please choose at least one layer above.")
    except URLError as e:
        st.error("""
            **This demo requires internet access.**

            Connection error: %s
        """ % e.reason)

df = pd.DataFrame({
    'first column': [1, 2, 3, 5],
    'second column': [12, 20, 30, 40]
})


@st.cache
def from_data_file(filename):
    url = (
        "http://raw.githubusercontent.com/streamlit/"
        "example-data/master/hello/v1/%s" % filename)
    return pd.read_json(url)


def display():
    pageHandlers = {
        "page1": lambda: st.write(df),
        "page2": lambda: st.write(df),
        "Geo Analysis": display_geo_analysis,
        "page4": lambda: st.write(df),
        "page5": lambda: st.write(df),
    }
    with st.sidebar:
        choose = option_menu("App Gallery", list(pageHandlers.keys()),
                             icons=['house', 'camera fill', 'kanban',
                                    'book', 'person lines fill'],
                             menu_icon="app-indicator", default_index=0,
                             styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#02ab21"},
        }
        )

    (pageHandlers[choose])()


def display_selectbox_style():
    """
    select box style page selection. NOT USING
    """
    # Create a page dropdown
    page = st.sidebar.selectbox(
        "Choose your page", ["Page 1", "Page 2", "Page 3"])
    if page == "Page 1":
        # Display details of page 1
        df
    elif page == "Page 2":
        # Display details of page 2
        df
    elif page == "Page 3":
        # Display details of page 3
        display_template_geodata()


if __name__ == '__main__':
    display()
