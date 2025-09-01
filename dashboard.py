import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

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

# --- Dataset info ---
st.subheader("📈 Dataset Info")
st.write(f"Total rows: {df.shape[0]}")
st.write(f"Total columns: {df.shape[1]}")

# --- Build graph if possible ---
if "source" in df.columns and "target" in df.columns:
    G = nx.from_pandas_edgelist(df, source="source", target="target")

    # --- Graph statistics ---
    st.subheader("📊 Graph Statistics")
    st.write(f"Number of nodes: {G.number_of_nodes()}")
    st.write(f"Number of edges: {G.number_of_edges()}")
    
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / len(degrees)
    st.write(f"Average degree: {avg_degree:.2f}")

    # --- Centrality Measures ---
    st.subheader("📌 Centrality Measures")
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    centrality_df = pd.DataFrame({
        "Node": list(G.nodes()),
        "Degree": [degrees[n] for n in G.nodes()],
        "Degree Centrality": [degree_centrality[n] for n in G.nodes()],
        "Betweenness": [betweenness[n] for n in G.nodes()],
        "Closeness": [closeness[n] for n in G.nodes()],
    }).sort_values(by="Degree", ascending=False)

    st.dataframe(centrality_df)

    # --- Filter by degree ---
    st.subheader("🎚️ Filter by Minimum Degree")
    min_degree = st.slider("Select minimum degree", 0, max(degrees.values()), 1)
    filtered_nodes = [n for n, d in degrees.items() if d >= min_degree]
    H = G.subgraph(filtered_nodes)

    # --- Interactive PyVis Graph ---
    st.subheader("🌐 Interactive Network Graph")
    net = Network(height="600px", width="100%", notebook=False)
    net.from_nx(H)
    net.show("network.html")
    st.components.v1.html(open("network.html", "r").read(), height=600)

    # --- Person search ---
    st.subheader("🔍 Search Connections")
    person = st.selectbox("Choose a person:", list(G.nodes()))
    neighbors = list(G.neighbors(person))
    st.write(f"Connections for {person}: {neighbors}")

    # --- Community detection ---
    st.subheader("🧩 Community Detection")
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    st.write(f"Detected {len(communities)} communities.")
    for i, c in enumerate(communities):
        st.write(f"Community {i+1}: {list(c)}")

    # --- Export filtered graph ---
    st.subheader("⬇️ Export Filtered Graph")
    export_df = nx.to_pandas_edgelist(H)
    st.download_button("Download CSV", export_df.to_csv(index=False), "filtered_graph.csv")

else:
    st.warning("⚠️ No 'source' and 'target' columns found. Skipping network graph.")
