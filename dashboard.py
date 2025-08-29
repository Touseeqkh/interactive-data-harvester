import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="People Network Dashboard", layout="wide")

st.title("👥 People Network Dashboard")

# --- File uploader with fallback ---
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ File uploaded successfully!")
else:
    st.info("ℹ️ No file uploaded. Using default `all_people.csv`")
    df = pd.read_csv("all_people.csv")

# --- Show data preview ---
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# --- Basic statistics ---
st.subheader("📈 Dataset Info")
st.write(f"Total rows: {df.shape[0]}")
st.write(f"Total columns: {df.shape[1]}")

# --- Graph building (if 'source' and 'target' columns exist) ---
if "source" in df.columns and "target" in df.columns:
    st.subheader("🌐 Network Graph")

    G = nx.from_pandas_edgelist(df, source="source", target="target")

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=500, node_color="skyblue", font_size=8, edge_color="gray")
    st.pyplot(plt)

else:
    st.warning("⚠️ No 'source' and 'target' columns found. Skipping network graph.")

# --- Allow user to search for a person ---
if "source" in df.columns:
    st.subheader("🔍 Search Connections")
    person = st.selectbox("Choose a person:", df["source"].unique())

    connections = df[df["source"] == person]["target"].tolist()
    st.write(f"Connections for {person}: {connections}")
