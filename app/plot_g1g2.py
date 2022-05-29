import numbers
import streamlit as st
import pandas as pd
import numpy as np
import sys
import altair as alt


path = sys.argv[1]
# 'scrapeDF_646_Samsung.csv'
df = pd.read_csv(path,sep=',') 

a = list(df["time of creation"])
time = []
time_int = []
for i in a :
    time.append(i[:10])
    time_int.append(int("".join(i[:10].split('-'))))
    
df["time"] = time
df["time_int"] = time_int
df = df.sort_values(by="time_int" , ascending= True) 
print (df)
dgb = df.groupby(['time'])
time = set(list(df["time"]))

time_post = []
kinds = []
number = []

for t in time:
    temp_group = dgb.get_group(t)
    neg = temp_group.loc[df["sentiment"] == "negative"]
    pos = temp_group.loc[df["sentiment"] == "positive"]
    neu = temp_group.loc[df["sentiment"] == "neutral"]
    
    time_post.append(t)
    kinds.append("negative")
    number.append(len(neg))
    
    time_post.append(t)
    kinds.append("positive")
    number.append(len(pos))
    
    time_post.append(t)
    kinds.append("neutral")
    number.append(len(neu))
    
dff = pd.DataFrame()
dff["time_post"] = time_post
dff["kinds"] = kinds
dff["number"] = number
print(dff)

    # print(len(neg),len(pos),len(ne))
    

import altair as alt
# from vega_datasets import data

# source = data.iowa_electricity()
source = dff

alt.Chart(source, title="Iowa's renewable energy boom").mark_area().encode(
    x=alt.X(
        "year:T",
        title="Year"
    ),
    y=alt.Y(
        # "net_generation:Q",
        "number:Q",
        stack="normalize",
        title="percent",
        axis=alt.Axis(format=".0%"),
    ),
    color=alt.Color(
        "source:N",
        legend=alt.Legend(title="sentiment"),
    )
)

# df = pd.DataFrame(
#      np.random.randn(200, 3),
#      columns=['a', 'b', 'c'])

# c = alt.Chart(df).mark_circle().encode(
#      x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

# st.altair_chart(c, use_container_width=True)
    
# print(dgb.get_group('2022-05-27'))





