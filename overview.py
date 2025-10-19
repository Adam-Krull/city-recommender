import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv('labeled.csv', index_col=0)

st.title("US City Similarity")
st.header("Project Overview")
st.text("My wife and I currently live in Texas and dream of moving to a " \
"state with mountains. We picked four states of interest: Idaho, Wyoming, " \
"Colorado, and Utah. I collected US Census Bureau information about every city " \
"in these four states as well as our home state. Collected information includes " \
"population, percent of population employed in different sectors, median " \
"home price and rent, homeownership rate, and median household income.")
st.text("I clustered the cities using K-means and hierarchical clustering. " \
"After exploring both sets of clusters, I determined the hierarchical clusters " \
"separated the cities into more distinct groups.")
st.text("Please explore my data and findings using this Streamlit application! " \
"The clusters page allows you to explore both sets of clusters: K-means and " \
"hierarchical. You can see firsthand how the hierarchical clusters do a better " \
"job separating the cities in line with human understanding. The cosine similarity " \
"page lets you input a city you know (or let RNG take the wheel) and returns the " \
"three most similar cities in the dataset. This could help you find cities you may " \
"like in other states! Finally, take a look at the dashboard. You can get a nice feel " \
"for the data from these high-level visuals.")
st.text("Enjoy a look at the data below! All pages are built using the information you see.")
st.dataframe(df)