import pandas as pd
import streamlit as st

df = pd.read_csv("labeled.csv", index_col=0)
to_drop = ["State", "KMeans", "Hierarchical"]
pop_medians = df.drop(columns=to_drop)
agg_columns = pop_medians.columns
pop_medians = pop_medians.agg("median")

if "cluster_type" not in st.session_state:
    st.session_state["cluster_type"] = None

st.header("Cluster Exploration")
st.text("Explore the clusters I created by two methods: K-means and " \
"hierarchical clustering. Select the cluster creation method below. " \
"You will see general information about the created clusters. Selecting " \
"a specific cluster number will reveal detailed metrics about that cluster.")

left, right = st.columns(2)
if left.button("K-means", type="primary", width="stretch"):
    st.session_state["cluster_type"] = "KMeans"
if right.button("Hierarchical", type="primary", width="stretch"):
    st.session_state["cluster_type"] = "Hierarchical"

if st.session_state["cluster_type"]:
        st.dataframe(df[st.session_state["cluster_type"]].value_counts().sort_index())    

st.text("Which cluster would you like to explore? All metrics are compared to the population median.")

cluster = st.pills(
    "Cluster #",
    options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    selection_mode="single"
)

if st.session_state["cluster_type"]:
    cluster_df = df[df[st.session_state["cluster_type"]] == cluster]
    state_counts = cluster_df["State"].value_counts()
    state_counts.name = "Counts"
    cluster_medians = cluster_df.drop(columns=to_drop).agg("median").round(2)
    cluster_medians.name = "Cluster medians"
    diffs = (cluster_medians - pop_medians).round(1)
    pcts = (diffs / pop_medians * 100).round(1)
    st.metric("Population", cluster_medians["Population"], diffs["Population"], border=True)
    a, b = st.columns(2)
    c, d = st.columns(2)
    e, f = st.columns(2)
    g, h = st.columns(2)
    i, j = st.columns(2)
    cards = [a, b, c, d, e, f, g, h, i, j]
    for col, card in zip(agg_columns[1:], cards):
         card.metric(col, cluster_medians[col], diffs[col], border=True)
    st.dataframe(cluster_df)     