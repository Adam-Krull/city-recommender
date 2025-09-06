import numpy as np
import pandas as pd
import streamlit as st

rng = np.random.default_rng()

df = pd.read_csv('labeled.csv', index_col=0)
top_3 = np.load('top_3.npy')
pointer = rng.integers(top_3.shape[0])

st.title('City Similarity Search')
st.divider()
st.header('Project Overview')
st.text('')
st.dataframe(df)