import numpy as np
import pandas as pd
import streamlit as st

df = pd.read_csv("labeled.csv")
rng = np.random.default_rng()
top_3 = np.load('top_3.npy').astype(int)
pointer = int(rng.integers(top_3.shape[0]))

st.header("Cosine Similarity")
st.text("This page allows you to explore the three most similar " \
"cities to your selected one using cosine similarity. The tool may " \
"help you discover cities of interest for vacations or moving.")

city = st.selectbox(
    "Which city would you like to search?",
    df["City"],
    index=pointer,
    key="city_pointer"
)
selected = df[df["City"] == city]

a, b, c = st.columns(3)
a.metric("City", selected["City"].str.replace(r",.+$", "", regex=True).array[0], border=True)
b.metric("State", selected["State"].array[0], border=True)
c.metric("Population", selected["Population"], border=True)

st.text(f"Three most similar cities to {selected['City'].array[0]}:")
d, e, f = st.columns(3)
indices = top_3[selected.index.array[0]]
top_3_info = df.iloc[indices]
for row, card in zip(top_3_info.iterrows(), [d, e, f]):
    card.metric(row[1]["City"], row[1]["Population"], border=True)